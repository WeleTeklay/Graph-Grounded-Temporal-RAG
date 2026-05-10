#accuracy_evaluator.py
"""
Accuracy evaluation suite for Graph-Grounded Temporal RAG.
"""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
from typing import List, Dict, Any
from dataclasses import dataclass

from src.query_engine import QueryEngine
from rich.console import Console
from rich.table import Table
console = Console()


@dataclass
class TestCase:
    question: str
    expected_answer: str
    target_date: str = None
    category: str = "general"


class AccuracyEvaluator:
    def __init__(self):
        self.engine = QueryEngine()
        self.results = []
    
    def evaluate(self, test_cases: List[TestCase]) -> Dict[str, Any]:
        """Run evaluation on test cases."""
        console.print("[bold cyan]🧪 Running Accuracy Evaluation...[/bold cyan]\n")
        
        for case in test_cases:
            start = time.time()
            result = self.engine.answer(case.question, case.target_date)
            elapsed = time.time() - start
            
            # Simple exact match (can be enhanced with ROUGE/BLEU)
            expected_lower = case.expected_answer.lower()
            actual_lower = result['answer'].lower()
            
            exact_match = expected_lower in actual_lower
            
            self.results.append({
                'category': case.category,
                'question': case.question,
                'expected': case.expected_answer,
                'actual': result['answer'][:200],
                'exact_match': exact_match,
                'graph_used': result['graph_used'],
                'time': elapsed,
                'sources': len(result['sources'])
            })
            
            icon = "✅" if exact_match else "❌"
            console.print(f"{icon} {case.question[:50]}... ({elapsed:.2f}s)")
        
        return self._generate_report()
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate evaluation report."""
        total = len(self.results)
        correct = sum(1 for r in self.results if r['exact_match'])
        accuracy = correct / total if total > 0 else 0
        
        # By category
        categories = {}
        for r in self.results:
            cat = r['category']
            if cat not in categories:
                categories[cat] = {'total': 0, 'correct': 0}
            categories[cat]['total'] += 1
            if r['exact_match']:
                categories[cat]['correct'] += 1
        
        # Display table
        table = Table(title="Evaluation Results", title_style="bold cyan")
        table.add_column("Category", style="cyan")
        table.add_column("Total", justify="right")
        table.add_column("Correct", justify="right", style="green")
        table.add_column("Accuracy", justify="right", style="yellow")
        
        for cat, stats in categories.items():
            acc = stats['correct'] / stats['total'] * 100
            table.add_row(cat, str(stats['total']), str(stats['correct']), f"{acc:.1f}%")
        
        table.add_row("─" * 20, "─" * 6, "─" * 7, "─" * 6)
        table.add_row("[bold]TOTAL[/bold]", str(total), str(correct), f"{accuracy*100:.1f}%")
        
        console.print(table)
        
        # Average time
        avg_time = sum(r['time'] for r in self.results) / total
        console.print(f"\n[dim]Average response time: {avg_time:.2f}s[/dim]")
        console.print(f"[dim]Graph enabled: {self.engine.graph_enabled}[/dim]")
        
        return {
            'accuracy': accuracy,
            'total': total,
            'correct': correct,
            'by_category': categories,
            'avg_time': avg_time
        }


def load_test_cases(file_path: str = "tests/test_cases.json") -> List[TestCase]:
    """Load test cases from JSON file."""
    path = Path(file_path)
    if not path.exists():
        console.print(f"[yellow]Warning: {file_path} not found. Using default test cases.[/yellow]")
        return get_default_test_cases()
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    return [TestCase(**case) for case in data]


def get_default_test_cases() -> List[TestCase]:
    """Return default test cases."""
    return [
        TestCase(
            question="What is the effective date of the original agreement?",
            expected_answer="2020-01-15",
            category="temporal"
        ),
        TestCase(
            question="What is the governing law?",
            expected_answer="Ethiopian",
            category="general"
        ),
        TestCase(
            question="Who are the parties?",
            expected_answer="Employer",
            category="general"
        ),
    ]


if __name__ == "__main__":
    evaluator = AccuracyEvaluator()
    test_cases = load_test_cases()
    report = evaluator.evaluate(test_cases)
    
    # Save report
    with open("evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    console.print("\n[green] Report saved to evaluation_report.json[/green]")