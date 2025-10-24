import json
from typing import List, Dict
from pathlib import Path
from datetime import datetime

class MetricsReporter:
    """Generate reports from evaluation results"""
    
    @staticmethod
    def generate_report(results: List[Dict], output_path: str = None):
        """Generate comprehensive evaluation report"""
        total_tests = len(results)
        
        # Aggregate metrics
        metric_sums = {}
        metric_counts = {}
        
        for result in results:
            for metric, value in result.get("metrics", {}).items():
                if isinstance(value, (int, float)):
                    metric_sums[metric] = metric_sums.get(metric, 0) + value
                    metric_counts[metric] = metric_counts.get(metric, 0) + 1
        
        # Calculate averages
        metric_averages = {
            metric: metric_sums[metric] / metric_counts[metric]
            for metric in metric_sums
        }
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": total_tests,
            "average_metrics": metric_averages,
            "detailed_results": results
        }
        
        # Save report
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report
    
    @staticmethod
    def print_summary(results: List[Dict]):
        """Print summary to console"""
        report = MetricsReporter.generate_report(results)
        
        print("\n" + "="*50)
        print("RAG EVALUATION SUMMARY")
        print("="*50)
        print(f"Total Tests: {report['total_tests']}")
        print("\nAverage Metrics:")
        for metric, value in report['average_metrics'].items():
            print(f"  {metric}: {value:.3f}")
        print("="*50 + "\n")