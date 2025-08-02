#!/usr/bin/env python3
"""
Complete experiment runner for GRPO vs SRRL comparison.
Trains both methods, evaluates them, and outputs JSON results.
"""

import argparse
import subprocess
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

def modify_training_script(method: str, config: dict, output_dir: str):
    """Create a modified training script for the specified method."""
    script_name = f"train_{method.lower()}_experiment.py"
    
    # Read the base training script
    with open("train_srrl.py", "r") as f:
        content = f.read()
    
    # Set SRRL flag
    use_srrl = method.upper() == "SRRL"
    content = content.replace("use_srrl = True", f"use_srrl = {use_srrl}")
    content = content.replace("use_srrl = False", f"use_srrl = {use_srrl}")
    
    # Set output directory
    content = content.replace('checkpoint_path = Path("./output")', f'checkpoint_path = Path("{output_dir}")')
    
    # Apply configuration overrides
    for key, value in config.items():
        import re
        pattern = rf"{key}\s*=\s*[^#\n]+"
        
        if isinstance(value, str):
            replacement = f'{key} = "{value}"'
        elif isinstance(value, bool):
            replacement = f'{key} = {value}'
        elif isinstance(value, (int, float)):
            replacement = f'{key} = {value}'
        else:
            replacement = f'{key} = {repr(value)}'
            
        content = re.sub(pattern, replacement, content)
    
    # Write the modified script
    with open(script_name, "w") as f:
        f.write(content)
    
    return script_name

def run_training(method: str, config: dict):
    """Train a model using the specified method."""
    print(f"\n{'='*60}")
    print(f"TRAINING {method.upper()} MODEL")
    print(f"{'='*60}")
    
    output_dir = f"./models/{method.lower()}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create training script
    script_name = modify_training_script(method, config, output_dir)
    
    try:
        # Run training
        print(f"Starting {method} training...")
        result = subprocess.run([sys.executable, script_name], check=True, capture_output=True, text=True)
        
        print(f"✅ {method} training completed successfully!")
        print(f"Model saved to: {output_dir}")
        
        # Clean up temporary script
        Path(script_name).unlink()
        
        return output_dir
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {method} training failed!")
        print(f"Error output: {e.stderr}")
        
        # Clean up temporary script
        if Path(script_name).exists():
            Path(script_name).unlink()
            
        raise e

def run_evaluation(grpo_model_path: str, srrl_model_path: str, config: dict):
    """Run evaluation comparing both models."""
    print(f"\n{'='*60}")
    print("RUNNING EVALUATION")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, "evaluate.py",
        "--grpo_model", grpo_model_path,
        "--srrl_model", srrl_model_path,
        "--test_size", str(config.get("test_size", 50)),
        "--num_samples", str(config.get("num_samples", 5))
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Evaluation completed successfully!")
        print(result.stdout)
        
        # Check if results file was created
        results_file = Path("evaluation_results.json")
        if results_file.exists():
            print(f"📊 Results saved to: {results_file}")
            
            # Load and print summary
            with open(results_file, "r") as f:
                results = json.load(f)
            
            print(f"\n🎯 FINAL RESULTS SUMMARY:")
            print(f"GRPO Pass@1: {results['grpo']['pass_at_1']:.3f}")
            print(f"SRRL Pass@1: {results['srrl']['pass_at_1']:.3f}")
            print(f"GRPO Pass@{config.get('num_samples', 5)}: {results['grpo']['pass_at_k']:.3f}")
            print(f"SRRL Pass@{config.get('num_samples', 5)}: {results['srrl']['pass_at_k']:.3f}")
            
            # Determine winner
            grpo_score = results['grpo']['pass_at_k']
            srrl_score = results['srrl']['pass_at_k']
            
            if srrl_score > grpo_score:
                improvement = (srrl_score - grpo_score) / grpo_score * 100 if grpo_score > 0 else 0
                print(f"🏆 SRRL WINS! (+{improvement:.1f}% improvement)")
            elif grpo_score > srrl_score:
                decline = (grpo_score - srrl_score) / grpo_score * 100 if grpo_score > 0 else 0
                print(f"📊 GRPO WINS! (SRRL is -{decline:.1f}% worse)")
            else:
                print(f"🤝 TIE! Both methods perform equally")
        
        return results_file
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed!")
        print(f"Error output: {e.stderr}")
        raise e

def main():
    parser = argparse.ArgumentParser(description="Run complete GRPO vs SRRL experiment")
    parser.add_argument("--mode", choices=["train", "eval", "both"], default="both",
                       help="Run training, evaluation, or both")
    parser.add_argument("--max_problems", type=int, default=10,
                       help="Number of training problems")
    parser.add_argument("--test_size", type=int, default=20,
                       help="Number of test problems for evaluation")
    parser.add_argument("--num_samples", type=int, default=5,
                       help="Number of samples per problem for evaluation")
    parser.add_argument("--group_size", type=int, default=4,
                       help="Group size for training")
    parser.add_argument("--rollouts_per_step", type=int, default=10,
                       help="Number of rollouts per training step")
    parser.add_argument("--lr", type=float, default=1e-5,
                       help="Learning rate")
    
    args = parser.parse_args()
    
    print("🚀 GRPO vs SRRL Experiment Runner")
    print(f"Mode: {args.mode}")
    
    # Configuration for both training runs
    config = {
        "max_problems": args.max_problems,
        "group_size": args.group_size,
        "rollouts_per_step": args.rollouts_per_step,
        "lr": args.lr,
        "test_size": args.test_size,
        "num_samples": args.num_samples,
        "wandb_project": None  # Disable wandb for clean runs
    }
    
    print(f"Configuration: {config}")
    
    grpo_model_path = None
    srrl_model_path = None
    
    # Training phase
    if args.mode in ["train", "both"]:
        print(f"\n📚 Starting training phase...")
        
        try:
            # Train GRPO
            grpo_model_path = run_training("GRPO", config)
            
            # Train SRRL
            srrl_model_path = run_training("SRRL", config)
            
            print(f"\n✅ Training phase completed!")
            print(f"GRPO model: {grpo_model_path}")
            print(f"SRRL model: {srrl_model_path}")
            
        except Exception as e:
            print(f"❌ Training phase failed: {e}")
            if args.mode == "both":
                print("Cannot proceed to evaluation without trained models.")
                return 1
            raise e
    
    # Evaluation phase
    if args.mode in ["eval", "both"]:
        # Use provided paths or default paths from training
        if grpo_model_path is None:
            grpo_model_path = "./models/grpo"
        if srrl_model_path is None:
            srrl_model_path = "./models/srrl"
            
        print(f"\n🔬 Starting evaluation phase...")
        
        try:
            results_file = run_evaluation(grpo_model_path, srrl_model_path, config)
            print(f"\n🎉 Experiment completed successfully!")
            print(f"📄 Full results available in: {results_file}")
            
        except Exception as e:
            print(f"❌ Evaluation phase failed: {e}")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 