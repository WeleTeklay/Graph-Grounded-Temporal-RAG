#src/streamlit run app.py
import json
from src.query_engine import QueryEngine

class RAGEvaluator:
    def __init__(self, engine):
        self.engine = engine
    
    def evaluate_answer(self, question: str, expected_answer: str):
        result = self.engine.answer(question)
        actual = result.get("answer", "")
        
        exact_match = expected_answer.lower() in actual.lower()
        
        return {
            "question": question,
            "expected": expected_answer,
            "actual": actual,
            "exact_match": exact_match
        }
    
    def run_test_suite(self, test_file: str = "tests/test_questions.json"):
        with open(test_file, 'r') as f:
            test_cases = json.load(f)
        
        results = []
        for tc in test_cases:
            result = self.evaluate_answer(tc['question'], tc['answer'])
            results.append(result)
            
            print("✓" if result['exact_match'] else "✗", tc['question'])
        
        accuracy = sum(r['exact_match'] for r in results) / len(results)
        print(f"\nAccuracy: {accuracy:.2%}")
        
        return results


if __name__ == "__main__":
    engine = QueryEngine()
    evaluator = RAGEvaluator(engine)
    evaluator.run_test_suite()