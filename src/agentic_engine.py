#src/agentic_engine.py
"""
Agentic Reasoning Layer for Graph-Grounded Temporal RAG.
Adds multi-step reasoning, self-reflection, and tool selection.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import ollama

from src.query_engine import QueryEngine
from src.logger import get_logger
from src.utils import handle_errors, Timer

logger = get_logger(__name__)


class ActionType(Enum):
    """Types of actions the agent can take."""
    RETRIEVE_CURRENT = "retrieve_current"
    RETRIEVE_HISTORICAL = "retrieve_historical"
    DETECT_CONTRADICTIONS = "detect_contradictions"
    COMPARE_VERSIONS = "compare_versions"
    SUMMARIZE = "summarize"
    VALIDATE = "validate"
    ANSWER = "answer"


@dataclass
class Thought:
    """A single reasoning step."""
    step: int
    action: ActionType
    reasoning: str
    result: Any
    confidence: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AgentPlan:
    """Plan for answering a complex question."""
    original_question: str
    sub_questions: List[str]
    actions: List[ActionType]
    thoughts: List[Thought] = field(default_factory=list)
    final_answer: Optional[str] = None
    confidence: float = 0.0
    validation_passed: bool = False


class AgenticReasoner:
    """
    Agentic reasoning layer that adds multi-step planning and self-reflection.
    """
    
    def __init__(self, query_engine: QueryEngine = None):
        self.engine = query_engine or QueryEngine()
        self.model = self.engine.config.ollama.model
        self.conversation_history: List[Dict] = []
        
    def _llm_think(self, prompt: str) -> str:
        """Use LLM for reasoning and planning."""
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={"temperature": 0.1, "num_predict": 500}
        )
        return response['response'].strip()
    
    def plan(self, question: str) -> AgentPlan:
        """
        Create a multi-step plan for answering a complex question.
        """
        planning_prompt = f"""[INST] You are an AI planning agent. Break down this legal question into sub-questions and actions.

Question: {question}

Available actions:
- retrieve_current: Get current version of relevant clauses
- retrieve_historical: Get historical versions
- detect_contradictions: Find conflicts between versions
- compare_versions: Compare old vs new language
- summarize: Synthesize findings
- validate: Check answer accuracy against sources

Output a JSON plan:
{{
    "sub_questions": ["question 1", "question 2"],
    "actions": ["action1", "action2", "action3"],
    "reasoning": "Why this plan makes sense"
}}

Output ONLY valid JSON. [/INST]"""
        
        try:
            response = self._llm_think(planning_prompt)
            # Extract JSON
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group(0))
            else:
                # Fallback plan
                plan_data = {
                    "sub_questions": [question],
                    "actions": ["retrieve_current", "answer"],
                    "reasoning": "Default plan"
                }
            
            return AgentPlan(
                original_question=question,
                sub_questions=plan_data.get('sub_questions', [question]),
                actions=[ActionType(a) for a in plan_data.get('actions', ['retrieve_current'])]
            )
        except Exception as e:
            logger.warning(f"Planning failed: {e}, using default plan")
            return AgentPlan(
                original_question=question,
                sub_questions=[question],
                actions=[ActionType.RETRIEVE_CURRENT, ActionType.ANSWER]
            )
    
    def execute_action(self, action: ActionType, context: Dict) -> Any:
        """Execute a single action."""
        
        if action == ActionType.RETRIEVE_CURRENT:
            return self.engine.answer(context['question'])
        
        elif action == ActionType.RETRIEVE_HISTORICAL:
            return self.engine.answer(
                context['question'],
                target_date=context.get('target_date', '2021-01-01')
            )
        
        elif action == ActionType.DETECT_CONTRADICTIONS:
            if hasattr(self.engine, 'answer_with_contradiction_awareness'):
                return self.engine.answer_with_contradiction_awareness(context['question'])
            return self.engine.answer(context['question'])
        
        elif action == ActionType.COMPARE_VERSIONS:
            current = self.engine.answer(context['question'])
            historical = self.engine.answer(context['question'], target_date='2021-01-01')
            return {
                'current': current,
                'historical': historical,
                'changed': current['answer'] != historical['answer']
            }
        
        elif action == ActionType.SUMMARIZE:
            sources = context.get('sources', [])
            summary_prompt = f"[INST] Summarize these legal findings in 3 bullet points:\n{sources}\n[/INST]"
            return self._llm_think(summary_prompt)
        
        elif action == ActionType.VALIDATE:
            return self.validate_answer(context['answer'], context.get('sources', []))
        
        elif action == ActionType.ANSWER:
            return context.get('accumulated_knowledge', '')
        
        return None
    
    def think(self, plan: AgentPlan) -> AgentPlan:
        """Execute the plan step by step, recording thoughts."""
        
        accumulated_context = {'question': plan.original_question}
        
        for i, action in enumerate(plan.actions, 1):
            logger.info(f"Agent step {i}: {action.value}")
            
            # Reasoning prompt
            reason_prompt = f"""[INST] You are executing step {i} of a plan to answer: "{plan.original_question}"
Current action: {action.value}
Context so far: {json.dumps(accumulated_context, default=str)[:500]}

What information are you seeking in this step? Respond in 1 sentence. [/INST]"""
            
            reasoning = self._llm_think(reason_prompt)
            
            # Execute action
            with Timer(f"Agent action: {action.value}"):
                result = self.execute_action(action, accumulated_context)
            
            # Update context
            if isinstance(result, dict):
                accumulated_context.update(result)
            else:
                accumulated_context[f'step_{i}_result'] = str(result)[:500]
            
            # Record thought
            thought = Thought(
                step=i,
                action=action,
                reasoning=reasoning,
                result=result,
                confidence=0.8  # Could be calculated
            )
            plan.thoughts.append(thought)
        
        # Store accumulated knowledge
        plan.final_answer = accumulated_context.get('answer', 
                          accumulated_context.get('accumulated_knowledge', ''))
        
        return plan
    
    def validate_answer(self, answer: str, sources: List[Dict]) -> Dict[str, Any]:
        """
        Self-reflection: Validate that the answer is grounded in sources.
        """
        validation_prompt = f"""[INST] You are validating an answer against its sources.

Answer: {answer[:500]}

Sources: {json.dumps([s.get('doc_id', '') for s in sources[:3]])}

Check:
1. Is the answer fully supported by sources?
2. Are there any hallucinations?
3. Is the temporal context correct?

Respond with JSON:
{{
    "grounded": true/false,
    "hallucinations": ["any unsupported claims"],
    "confidence": 0.0-1.0,
    "explanation": "brief validation note"
}} [/INST]"""
        
        try:
            response = self._llm_think(validation_prompt)
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except:
            pass
        
        return {"grounded": True, "confidence": 0.7, "explanation": "Validation passed"}
    
    def reflect_and_refine(self, plan: AgentPlan) -> AgentPlan:
        """
        Self-reflection: Check if answer is sufficient, refine if needed.
        """
        if not plan.final_answer:
            return plan
        
        # Validate
        sources = []
        for thought in plan.thoughts:
            if isinstance(thought.result, dict) and 'sources' in thought.result:
                sources.extend(thought.result['sources'])
        
        validation = self.validate_answer(plan.final_answer, sources)
        plan.validation_passed = validation.get('grounded', True)
        plan.confidence = validation.get('confidence', 0.7)
        
        # Refine if needed
        if not plan.validation_passed or plan.confidence < 0.6:
            refine_prompt = f"""[INST] The previous answer may be incomplete. 
Question: {plan.original_question}
Previous answer: {plan.final_answer}
Validation: {validation}

Provide a refined, more accurate answer. [/INST]"""
            
            plan.final_answer = self._llm_think(refine_prompt)
            plan.confidence = min(plan.confidence + 0.2, 1.0)
        
        return plan
    
    def answer_with_agent(self, question: str) -> Dict[str, Any]:
        """
        Complete agentic pipeline: Plan → Execute → Reflect → Answer.
        """
        logger.info(f"Agent processing: {question}")
        
        # Step 1: Plan
        plan = self.plan(question)
        logger.info(f"Plan created: {len(plan.actions)} actions")
        
        # Step 2: Execute (Think)
        plan = self.think(plan)
        
        # Step 3: Reflect and Refine
        plan = self.reflect_and_refine(plan)
        
        # Step 4: Build response
        return {
            'answer': plan.final_answer or "Unable to answer",
            'confidence': plan.confidence,
            'validation_passed': plan.validation_passed,
            'reasoning_steps': [
                {
                    'step': t.step,
                    'action': t.action.value,
                    'reasoning': t.reasoning
                }
                for t in plan.thoughts
            ],
            'sub_questions': plan.sub_questions,
            'agent_used': True
        }
    
    def chat(self, message: str) -> str:
        """Conversational interface with memory."""
        # Add to history
        self.conversation_history.append({"role": "user", "content": message})
        
        # Build context
        context = "\n".join([
            f"{h['role']}: {h['content']}" 
            for h in self.conversation_history[-5:]
        ])
        
        prompt = f"""[INST] You are a legal assistant with memory of the conversation.

Conversation history:
{context}

Current question: {message}

Answer based on the conversation context and your knowledge. [/INST]"""
        
        response = self._llm_think(prompt)
        
        # Add to history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response


class AgenticQueryEngine(QueryEngine):
    """
    Enhanced QueryEngine with agentic capabilities.
    """
    
    def __init__(self):
        super().__init__()
        self.agent = AgenticReasoner(self)
        self.use_agent = True
    
    def answer(self, question: str, target_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Override answer method to use agent for complex questions.
        """
        # Detect if question is complex (multiple parts, comparisons, etc.)
        complex_indicators = ['compare', 'difference', 'change', 'evolve', 'history', 'before and after', 'what if']
        
        is_complex = any(ind in question.lower() for ind in complex_indicators)
        is_complex = is_complex or '?' in question[question.find('?'):]  # Multiple questions
        
        if self.use_agent and is_complex:
            logger.info(f"Using agent for complex question: {question[:50]}...")
            result = self.agent.answer_with_agent(question)
            result['graph_used'] = self.graph_enabled
            result['agent_used'] = True
            return result
        else:
            result = super().answer(question, target_date)
            result['agent_used'] = False
            return result
# Add to src/agentic_engine.py

class SelfCorrectingRAG(AgenticReasoner):
    """RAG that verifies and corrects its own answers."""
    
    def verify_and_correct(self, question: str, initial_answer: str, sources: List[Dict]) -> Dict[str, Any]:
        """
        Verify answer against sources, correct if hallucinated.
        """
        verification_prompt = f"""[INST] You are a strict legal fact-checker.

Question: {question}
Proposed Answer: {initial_answer}
Sources: {json.dumps([{'doc': s.get('doc_id'), 'date': s.get('effective_date')} for s in sources[:5]])}

Task:
1. Identify any claims NOT supported by sources
2. Identify any contradictions with sources
3. Provide a corrected answer using ONLY source information

Output JSON:
{{
    "hallucinations": ["claim1", "claim2"],
    "contradictions": ["issue1"],
    "corrected_answer": "factual answer based only on sources",
    "correction_needed": true/false,
    "confidence": 0.0-1.0
}} [/INST]"""
        
        response = self._llm_think(verification_prompt)
        
        try:
            import re
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                return result
        except:
            pass
        
        return {
            "hallucinations": [],
            "corrected_answer": initial_answer,
            "correction_needed": False,
            "confidence": 0.8
        }
    
    def iterative_retrieval(self, question: str, max_iterations: int = 3) -> Dict[str, Any]:
        """
        Retrieve, verify, and iteratively improve until confidence threshold met.
        """
        confidence_threshold = 0.85
        current_question = question
        all_sources = []
        
        for iteration in range(max_iterations):
            # Retrieve
            result = self.engine.answer(current_question)
            
            if not result.get('sources'):
                break
            
            all_sources.extend(result['sources'])
            
            # Verify
            verification = self.verify_and_correct(
                question, 
                result['answer'], 
                all_sources
            )
            
            if verification.get('confidence', 0) >= confidence_threshold:
                return {
                    'answer': verification.get('corrected_answer', result['answer']),
                    'sources': all_sources,
                    'iterations': iteration + 1,
                    'confidence': verification['confidence'],
                    'hallucinations_detected': verification.get('hallucinations', []),
                    'self_corrected': verification.get('correction_needed', False)
                }
            
            # Refine question for next iteration
            if verification.get('hallucinations'):
                current_question = f"{question} (specifically about: {', '.join(verification['hallucinations'][:2])})"
        
        return {
            'answer': result['answer'],
            'sources': all_sources,
            'iterations': max_iterations,
            'confidence': verification.get('confidence', 0.5),
            'max_iterations_reached': True
        }
# Add to src/agentic_engine.py

class ComparativeAnalyzer:
    """Analyze and compare clauses across document versions."""
    
    def __init__(self, engine):
        self.engine = engine
        self.model = engine.config.ollama.model
    
    def compare_across_versions(self, topic: str, doc_ids: List[str]) -> Dict[str, Any]:
        """
        Compare how a topic evolved across specific document versions.
        """
        results = {}
        
        for doc_id in doc_ids:
            query = f"What does {doc_id} say about {topic}?"
            result = self.engine.answer(query)
            results[doc_id] = {
                'answer': result['answer'],
                'sources': result.get('sources', [])
            }
        
        # Generate comparison summary
        comparison_prompt = f"""[INST] Compare how "{topic}" evolved across these documents:

{json.dumps({k: v['answer'][:300] for k, v in results.items()}, indent=2)}

Provide:
1. Timeline of changes
2. Key differences between versions
3. Current effective version
4. Any contradictions resolved

Output as structured analysis. [/INST]"""
        
        response = ollama.generate(
            model=self.model,
            prompt=comparison_prompt,
            options={"temperature": 0.1, "num_predict": 600}
        )
        
        return {
            'topic': topic,
            'version_results': results,
            'comparative_analysis': response['response'].strip(),
            'evolution_detected': len(set(v['answer'][:100] for v in results.values())) > 1
        }
    
    def find_all_amendments(self, base_doc_id: str) -> List[Dict]:
        """Trace all amendments to a base document."""
        amendments = []
        
        if self.engine.graph_enabled:
            with self.engine.neo4j_driver.session() as session:
                result = session.run("""
                    MATCH path = (base:Clause {doc_id: $doc_id})-[:SUPERSEDES*]->(amendment:Clause)
                    RETURN amendment.doc_id as doc_id, 
                           amendment.effective_date as date,
                           length(path) as depth
                    ORDER BY amendment.effective_date
                """, doc_id=base_doc_id)
                
                for record in result:
                    amendments.append({
                        'doc_id': record['doc_id'],
                        'effective_date': str(record['date']),
                        'amendment_depth': record['depth']
                    })
        
        return amendments
# Add to src/agentic_engine.py

class ContradictionMonitor:
    """Proactively monitor and alert on document contradictions."""
    
    def __init__(self, engine):
        self.engine = engine
        self.detected_contradictions = []
    
    def scan_all_clauses(self) -> List[Dict]:
        """Scan entire document set for contradictions."""
        contradictions = []
        
        if not self.engine.graph_enabled:
            return contradictions
        
        with self.engine.neo4j_driver.session() as session:
            # Find clauses with multiple versions
            result = session.run("""
                MATCH (c1:Clause)
                MATCH (c2:Clause)
                WHERE c1.doc_id < c2.doc_id
                  AND c1.text CONTAINS c2.text
                  AND c1.effective_date < c2.effective_date
                RETURN c1.id as old_id, c1.doc_id as old_doc, c1.text as old_text,
                       c2.id as new_id, c2.doc_id as new_doc, c2.text as new_text
                LIMIT 20
            """)
            
            for record in result:
                # Check if actually contradictory (not just updated)
                if self._is_contradictory(record['old_text'], record['new_text']):
                    contradictions.append({
                        'old_version': {
                            'doc_id': record['old_doc'],
                            'text': record['old_text'][:200]
                        },
                        'new_version': {
                            'doc_id': record['new_doc'],
                            'text': record['new_text'][:200]
                        },
                        'detected_at': datetime.now().isoformat()
                    })
        
        self.detected_contradictions = contradictions
        return contradictions
    
    def _is_contradictory(self, text1: str, text2: str) -> bool:
        """Use LLM to determine if texts contradict."""
        prompt = f"""[INST] Do these two legal clauses contradict each other?

Clause 1: {text1[:300]}
Clause 2: {text2[:300]}

Answer ONLY "yes" or "no". [/INST]"""
        
        response = ollama.generate(
            model=self.engine.config.ollama.model,
            prompt=prompt,
            options={"num_predict": 5}
        )
        
        return 'yes' in response['response'].lower()
    
    def generate_alert_report(self) -> str:
        """Generate human-readable contradiction report."""
        if not self.detected_contradictions:
            return "No contradictions detected in current document set."
        
        report = "## Contradiction Alert Report\n\n"
        
        for i, c in enumerate(self.detected_contradictions, 1):
            report += f"### Contradiction {i}\n"
            report += f"- **Old Version** ({c['old_version']['doc_id']}): {c['old_version']['text'][:100]}...\n"
            report += f"- **New Version** ({c['new_version']['doc_id']}): {c['new_version']['text'][:100]}...\n\n"
        
        return report
# Add to src/agentic_engine.py

class ScenarioAnalyzer:
    """Analyze hypothetical scenarios based on document evolution."""
    
    def __init__(self, engine):
        self.engine = engine
        self.model = engine.config.ollama.model
    
    def analyze_scenario(self, scenario: str, target_date: str = None) -> Dict[str, Any]:
        """
        Analyze a hypothetical scenario.
        Example: "What if this agreement was signed in 2021 instead of 2023?"
        """
        # Extract key elements from scenario
        extraction_prompt = f"""[INST] Extract key legal elements from this scenario:
"{scenario}"

Output JSON:
{{
    "topic": "main subject",
    "timeframe": "date mentioned",
    "condition": "what if condition",
    "comparison_baseline": "what to compare against"
}} [/INST]"""
        
        response = self._llm_think(extraction_prompt)
        
        try:
            import re
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            elements = json.loads(json_match.group(0)) if json_match else {}
        except:
            elements = {"topic": scenario, "timeframe": None}
        
        # Get current state
        current = self.engine.answer(
            f"What is the current rule about {elements.get('topic', scenario)}?",
            target_date=target_date
        )
        
        # Get historical state if timeframe specified
        historical = None
        if elements.get('timeframe'):
            historical = self.engine.answer(
                f"What was the rule about {elements.get('topic', scenario)}?",
                target_date=elements['timeframe']
            )
        
        # Generate scenario analysis
        analysis_prompt = f"""[INST] Analyze this legal hypothetical scenario:
{scenario}

Current rule: {current['answer']}
{f"Historical rule ({elements.get('timeframe')}): {historical['answer']}" if historical else ""}

Provide:
1. What would be different under this scenario
2. Legal implications
3. Recommended actions

Be precise and cite sources. [/INST]"""
        
        analysis = self._llm_think(analysis_prompt)
        
        return {
            'scenario': scenario,
            'current_state': current,
            'historical_state': historical,
            'analysis': analysis,
            'elements_extracted': elements
        }

# Add to src/agentic_engine.py

class ChainOfVerificationRAG:
    """RAG with chain-of-verification for high-stakes answers."""
    
    def __init__(self, engine):
        self.engine = engine
    
    def answer_with_verification_chain(self, question: str) -> Dict[str, Any]:
        """
        Four-step verification chain:
        1. Initial retrieval
        2. Fact extraction
        3. Independent verification
        4. Final answer synthesis
        """
        
        # Step 1: Initial retrieval
        initial = self.engine.answer(question)
        
        # Step 2: Extract factual claims
        claims_prompt = f"""[INST] Extract all factual claims from this answer as a JSON list:
{initial['answer']}

Output: ["claim1", "claim2", ...] [/INST]"""
        
        claims_response = self._llm_think(claims_prompt)
        try:
            claims = json.loads(claims_response)
        except:
            claims = [initial['answer'][:100]]
        
        # Step 3: Verify each claim independently
        verified_claims = []
        for claim in claims[:5]:
            verification = self.engine.answer(f"Is this true according to the documents: {claim}")
            verified_claims.append({
                'claim': claim,
                'verification': verification['answer'],
                'verified': 'true' in verification['answer'].lower() or 'yes' in verification['answer'].lower()
            })
        
        # Step 4: Synthesize verified answer
        synthesis_prompt = f"""[INST] Synthesize a final answer using ONLY verified claims:
Original question: {question}
Verified claims: {json.dumps([c for c in verified_claims if c['verified']])}

Provide a concise, verified answer. [/INST]"""
        
        final_answer = self._llm_think(synthesis_prompt)
        
        return {
            'answer': final_answer,
            'initial_answer': initial['answer'],
            'verified_claims': verified_claims,
            'verification_passed': all(c['verified'] for c in verified_claims),
            'confidence': sum(c['verified'] for c in verified_claims) / len(verified_claims) if verified_claims else 0.5
        }
    
# Add to src/agentic_engine.py

class MasterLegalAgent:
    """
    Unified agent combining all RAG + Agentic capabilities.
    """
    
    def __init__(self):
        self.engine = AgenticQueryEngine()
        self.self_correcting = SelfCorrectingRAG(self.engine)
        self.comparative = ComparativeAnalyzer(self.engine)
        self.monitor = ContradictionMonitor(self.engine)
        self.scenario = ScenarioAnalyzer(self.engine)
        self.verification = ChainOfVerificationRAG(self.engine)
        
    def process(self, query: str, mode: str = "auto") -> Dict[str, Any]:
        """
        Process query with automatic mode selection.
        
        Modes:
        - auto: Intelligently select best approach
        - verify: Use chain-of-verification
        - compare: Multi-document comparison
        - scenario: Hypothetical analysis
        - monitor: Scan for contradictions
        """
        
        if mode == "auto":
            mode = self._select_mode(query)
        
        if mode == "verify":
            return self.verification.answer_with_verification_chain(query)
        elif mode == "compare":
            return self._handle_compare(query)
        elif mode == "scenario":
            return self.scenario.analyze_scenario(query)
        elif mode == "monitor":
            contradictions = self.monitor.scan_all_clauses()
            return {
                'mode': 'monitor',
                'contradictions_found': len(contradictions),
                'report': self.monitor.generate_alert_report(),
                'details': contradictions
            }
        else:
            # Default: use self-correcting RAG
            return self.self_correcting.iterative_retrieval(query)
    
    def _select_mode(self, query: str) -> str:
        """Intelligently select processing mode."""
        query_lower = query.lower()
        
        if any(w in query_lower for w in ['verify', 'check', 'confirm', 'true', 'correct']):
            return "verify"
        elif any(w in query_lower for w in ['compare', 'difference', 'versus', 'vs', 'change']):
            return "compare"
        elif any(w in query_lower for w in ['what if', 'scenario', 'hypothetical', 'suppose']):
            return "scenario"
        elif any(w in query_lower for w in ['contradiction', 'conflict', 'inconsistent']):
            return "monitor"
        else:
            return "default"
    
    def _handle_compare(self, query: str) -> Dict[str, Any]:
        """Handle comparison queries."""
        # Extract document IDs or topics
        topics = re.findall(r'"([^"]*)"', query)
        if len(topics) >= 2:
            return self.comparative.compare_across_versions(topics[0], topics[1:])
        return {'error': 'Please specify documents to compare in quotes'}
    
