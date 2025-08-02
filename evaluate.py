"""
Evaluation script for comparing GRPO vs SRRL models on MBPP test set.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mbpp_utils import (
    load_mbpp_dataset, 
    execute_code_with_tests, 
    calculate_code_reward,
    extract_code_from_completion
)


def load_trained_model(model_path: str, device: str = "cuda"):
    """Load a trained model from checkpoint."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


@torch.no_grad()
def generate_solution(model, tokenizer, prompt: str, max_length: int = 1024, temperature: float = 0.2, num_samples: int = 1):
    """Generate code solutions for a given prompt."""
    model.eval()
    
    # Format prompt
    if tokenizer.chat_template:
        chat_messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted_prompt = prompt
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Generate multiple samples
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,
        do_sample=True,
        num_return_sequences=num_samples,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    # Decode completions (excluding the input)
    completions = []
    for output in outputs:
        completion = tokenizer.decode(output[inputs.input_ids.shape[1]:], skip_special_tokens=True)
        completions.append(completion)
    
    return completions


def evaluate_model(model_path: str, test_problems: List[Dict], num_samples: int = 5, device: str = "cuda") -> Dict[str, Any]:
    """Evaluate a model on test problems."""
    print(f"Loading model from {model_path}")
    model, tokenizer = load_trained_model(model_path, device)
    
    results = {
        "model_path": model_path,
        "total_problems": len(test_problems),
        "success_count": 0,
        "pass_at_1": 0.0,
        "pass_at_k": 0.0,
        "avg_reward": 0.0,
        "problem_results": []
    }
    
    for i, problem in enumerate(test_problems):
        print(f"Evaluating problem {i+1}/{len(test_problems)}: {problem['task_id']}")
        
        # Generate solutions
        completions = generate_solution(
            model, tokenizer, problem["prompt"], 
            num_samples=num_samples
        )
        
        # Evaluate each solution
        problem_result = {
            "task_id": problem["task_id"],
            "prompt": problem["prompt"],
            "solutions": [],
            "best_reward": 0.0,
            "pass_at_1": False,
            "pass_at_k": False
        }
        
        rewards = []
        for j, completion in enumerate(completions):
            code = extract_code_from_completion(completion)
            execution_result = execute_code_with_tests(
                code, problem["test_list"], problem.get("test_imports", [])
            )
            reward = calculate_code_reward(execution_result)
            rewards.append(reward)
            
            solution_result = {
                "completion": completion,
                "code": code,
                "reward": reward,
                "success": execution_result["success"],
                "execution_result": execution_result
            }
            problem_result["solutions"].append(solution_result)
        
        # Calculate metrics for this problem
        problem_result["best_reward"] = max(rewards)
        problem_result["pass_at_1"] = rewards[0] >= 1.0  # First solution passes
        problem_result["pass_at_k"] = any(r >= 1.0 for r in rewards)  # Any solution passes
        
        results["problem_results"].append(problem_result)
        
        # Update global metrics
        if problem_result["pass_at_k"]:
            results["success_count"] += 1
    
    # Calculate final metrics
    results["pass_at_1"] = sum(p["pass_at_1"] for p in results["problem_results"]) / len(test_problems)
    results["pass_at_k"] = sum(p["pass_at_k"] for p in results["problem_results"]) / len(test_problems)
    results["avg_reward"] = sum(p["best_reward"] for p in results["problem_results"]) / len(test_problems)
    
    print(f"\nResults for {model_path}:")
    print(f"Pass@1: {results['pass_at_1']:.3f}")
    print(f"Pass@{num_samples}: {results['pass_at_k']:.3f}")
    print(f"Average Reward: {results['avg_reward']:.3f}")
    print(f"Success Rate: {results['success_count']}/{results['total_problems']}")
    
    return results


def compare_models(grpo_path: str, srrl_path: str, test_problems: List[Dict], num_samples: int = 5):
    """Compare GRPO vs SRRL models."""
    print("="*60)
    print("COMPARING GRPO vs SRRL MODELS")
    print("="*60)
    
    # Evaluate GRPO model
    print("\n" + "="*30)
    print("EVALUATING GRPO MODEL")
    print("="*30)
    grpo_results = evaluate_model(grpo_path, test_problems, num_samples)
    
    # Evaluate SRRL model
    print("\n" + "="*30)
    print("EVALUATING SRRL MODEL")
    print("="*30)
    srrl_results = evaluate_model(srrl_path, test_problems, num_samples)
    
    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    metrics = ["pass_at_1", "pass_at_k", "avg_reward"]
    
    for metric in metrics:
        grpo_val = grpo_results[metric]
        srrl_val = srrl_results[metric]
        improvement = (srrl_val - grpo_val) / grpo_val * 100 if grpo_val > 0 else 0
        
        print(f"{metric.upper()}:")
        print(f"  GRPO:  {grpo_val:.3f}")
        print(f"  SRRL:  {srrl_val:.3f}")
        print(f"  Improvement: {improvement:+.1f}%")
        print()
    
    # Save detailed results
    comparison_results = {
        "grpo": grpo_results,
        "srrl": srrl_results,
        "comparison": {
            "pass_at_1_improvement": (srrl_results["pass_at_1"] - grpo_results["pass_at_1"]) / grpo_results["pass_at_1"] * 100 if grpo_results["pass_at_1"] > 0 else 0,
            "pass_at_k_improvement": (srrl_results["pass_at_k"] - grpo_results["pass_at_k"]) / grpo_results["pass_at_k"] * 100 if grpo_results["pass_at_k"] > 0 else 0,
            "avg_reward_improvement": (srrl_results["avg_reward"] - grpo_results["avg_reward"]) / grpo_results["avg_reward"] * 100 if grpo_results["avg_reward"] > 0 else 0,
        }
    }
    
    # Save results to file
    results_file = Path("evaluation_results.json")
    with open(results_file, "w") as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"Detailed results saved to {results_file}")
    
    return comparison_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained GRPO and SRRL models")
    parser.add_argument("--grpo_model", type=str, required=True, help="Path to trained GRPO model")
    parser.add_argument("--srrl_model", type=str, required=True, help="Path to trained SRRL model")
    parser.add_argument("--test_size", type=int, default=100, help="Number of test problems to evaluate")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate per problem")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for evaluation")
    
    args = parser.parse_args()
    
    # Load test dataset
    print(f"Loading MBPP test dataset (max {args.test_size} problems)...")
    test_problems = load_mbpp_dataset("test", max_samples=args.test_size)
    print(f"Loaded {len(test_problems)} test problems")
    
    # Compare models
    results = compare_models(
        args.grpo_model, 
        args.srrl_model, 
        test_problems, 
        args.num_samples
    )
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    grpo_pass_k = results["grpo"]["pass_at_k"]
    srrl_pass_k = results["srrl"]["pass_at_k"]
    
    if srrl_pass_k > grpo_pass_k:
        print(f"🎉 SRRL outperforms GRPO by {srrl_pass_k - grpo_pass_k:.3f} points on Pass@{args.num_samples}")
    elif grpo_pass_k > srrl_pass_k:
        print(f"📊 GRPO outperforms SRRL by {grpo_pass_k - srrl_pass_k:.3f} points on Pass@{args.num_samples}")
    else:
        print(f"🤝 GRPO and SRRL perform equally on Pass@{args.num_samples}")


if __name__ == "__main__":
    main() 