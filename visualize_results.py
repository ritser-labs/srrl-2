#!/usr/bin/env python3
"""
Visualization script for GRPO vs SRRL experiment results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def load_results(results_file: str = "evaluation_results.json"):
    """Load evaluation results from JSON file."""
    if not Path(results_file).exists():
        print(f"Results file {results_file} not found. Run evaluation first.")
        return None
    
    with open(results_file, "r") as f:
        return json.load(f)


def plot_comparison_metrics(results):
    """Plot comparison metrics between GRPO and SRRL."""
    grpo = results["grpo"]
    srrl = results["srrl"]
    
    metrics = ["pass_at_1", "pass_at_k", "avg_reward"]
    grpo_values = [grpo[m] for m in metrics]
    srrl_values = [srrl[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, grpo_values, width, label='GRPO', alpha=0.8, color='skyblue')
    bars2 = ax.bar(x + width/2, srrl_values, width, label='SRRL', alpha=0.8, color='lightcoral')
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('GRPO vs SRRL Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '@').title() for m in metrics])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    plt.tight_layout()
    plt.savefig('grpo_vs_srrl_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_improvement_breakdown(results):
    """Plot improvement breakdown by metric."""
    comparison = results["comparison"]
    
    metrics = ["pass_at_1_improvement", "pass_at_k_improvement", "avg_reward_improvement"]
    improvements = [comparison[m] for m in metrics]
    
    colors = ['green' if x > 0 else 'red' for x in improvements]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(range(len(metrics)), improvements, color=colors, alpha=0.7)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Improvement (%)')
    ax.set_title('SRRL Improvement over GRPO')
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m.replace('_improvement', '').replace('_', '@').title() for m in metrics])
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, improvements)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
               f'{val:+.1f}%', ha='center', va='bottom' if height > 0 else 'top')
    
    plt.tight_layout()
    plt.savefig('srrl_improvement_breakdown.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_problem_wise_analysis(results):
    """Plot problem-wise success analysis."""
    grpo_problems = results["grpo"]["problem_results"]
    srrl_problems = results["srrl"]["problem_results"]
    
    # Ensure same ordering
    grpo_dict = {p["task_id"]: p for p in grpo_problems}
    srrl_dict = {p["task_id"]: p for p in srrl_problems}
    
    common_tasks = set(grpo_dict.keys()) & set(srrl_dict.keys())
    common_tasks = sorted(common_tasks)
    
    grpo_success = [grpo_dict[task]["pass_at_k"] for task in common_tasks]
    srrl_success = [srrl_dict[task]["pass_at_k"] for task in common_tasks]
    
    # Calculate improvement categories
    both_pass = sum(1 for g, s in zip(grpo_success, srrl_success) if g and s)
    only_grpo = sum(1 for g, s in zip(grpo_success, srrl_success) if g and not s)
    only_srrl = sum(1 for g, s in zip(grpo_success, srrl_success) if not g and s)
    both_fail = sum(1 for g, s in zip(grpo_success, srrl_success) if not g and not s)
    
    categories = ['Both Pass', 'Only GRPO', 'Only SRRL', 'Both Fail']
    counts = [both_pass, only_grpo, only_srrl, both_fail]
    colors = ['green', 'blue', 'orange', 'red']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pie chart
    ax1.pie(counts, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Problem-wise Success Distribution')
    
    # Bar chart
    ax2.bar(categories, counts, color=colors, alpha=0.7)
    ax2.set_ylabel('Number of Problems')
    ax2.set_title('Problem-wise Success Counts')
    ax2.grid(True, alpha=0.3)
    
    # Add count labels
    for i, count in enumerate(counts):
        ax2.text(i, count + 0.5, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('problem_wise_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_summary_stats(results):
    """Print detailed summary statistics."""
    print("\n" + "="*60)
    print("DETAILED SUMMARY STATISTICS")
    print("="*60)
    
    grpo = results["grpo"]
    srrl = results["srrl"]
    comparison = results["comparison"]
    
    print(f"\nGRPO Results:")
    print(f"  Total Problems: {grpo['total_problems']}")
    print(f"  Success Count: {grpo['success_count']}")
    print(f"  Pass@1: {grpo['pass_at_1']:.3f}")
    print(f"  Pass@k: {grpo['pass_at_k']:.3f}")
    print(f"  Average Reward: {grpo['avg_reward']:.3f}")
    
    print(f"\nSRRL Results:")
    print(f"  Total Problems: {srrl['total_problems']}")
    print(f"  Success Count: {srrl['success_count']}")
    print(f"  Pass@1: {srrl['pass_at_1']:.3f}")
    print(f"  Pass@k: {srrl['pass_at_k']:.3f}")
    print(f"  Average Reward: {srrl['avg_reward']:.3f}")
    
    print(f"\nImprovements (SRRL over GRPO):")
    print(f"  Pass@1: {comparison['pass_at_1_improvement']:+.1f}%")
    print(f"  Pass@k: {comparison['pass_at_k_improvement']:+.1f}%")
    print(f"  Average Reward: {comparison['avg_reward_improvement']:+.1f}%")
    
    # Statistical significance (simple)
    grpo_successes = grpo['success_count']
    srrl_successes = srrl['success_count']
    total_problems = grpo['total_problems']
    
    improvement = srrl_successes - grpo_successes
    print(f"\nAbsolute Improvement:")
    print(f"  Success Count: {improvement:+d} problems")
    print(f"  Success Rate: {improvement/total_problems:+.3f}")
    
    if improvement > 0:
        print(f"\n🎉 SRRL shows improvement over GRPO!")
    elif improvement < 0:
        print(f"\n📊 GRPO performs better than SRRL.")
    else:
        print(f"\n🤝 GRPO and SRRL perform equally.")


def main():
    parser = argparse.ArgumentParser(description="Visualize GRPO vs SRRL results")
    parser.add_argument("--results_file", type=str, default="evaluation_results.json",
                       help="Path to evaluation results JSON file")
    parser.add_argument("--no_plots", action="store_true",
                       help="Skip generating plots (useful for headless environments)")
    
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.results_file)
    if results is None:
        return
    
    # Print summary
    print_summary_stats(results)
    
    if not args.no_plots:
        try:
            # Generate plots
            print("\nGenerating visualizations...")
            plot_comparison_metrics(results)
            plot_improvement_breakdown(results)
            plot_problem_wise_analysis(results)
            print("✅ Visualizations saved as PNG files")
        except ImportError:
            print("⚠️  matplotlib not available. Install with: pip install matplotlib")
        except Exception as e:
            print(f"❌ Error generating plots: {e}")


if __name__ == "__main__":
    main() 