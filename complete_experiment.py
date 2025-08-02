#!/usr/bin/env python3
"""
Complete GRPO vs SRRL Experiment
- Trains both GRPO and SRRL models with on-policy RL
- Evaluates both models on unseen test problems  
- Saves comprehensive results with training metrics AND post-training accuracy
"""

import os
import sys
import json
import torch
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import traceback

# Import our modules
from train_srrl import main as train_main
from evaluate import compare_models
from mbpp_utils import load_mbpp_dataset

def modify_and_run_training(method: str, output_dir: str, max_problems: int = 6) -> str:
    """Modify training script and run training for specified method."""
    print(f"\n{'='*60}")
    print(f"TRAINING {method.upper()} MODEL")
    print(f"{'='*60}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read the training script
    with open("train_srrl.py", "r") as f:
        content = f.read()
    
    # Set method flag
    use_srrl = method.upper() == "SRRL"
    content = content.replace("use_srrl = True", f"use_srrl = {use_srrl}")
    content = content.replace("use_srrl = False", f"use_srrl = {use_srrl}")
    
    # Set output directory
    content = content.replace('checkpoint_path = Path("./output")', f'checkpoint_path = Path("{output_dir}")')
    
    # Configure for experiment
    content = content.replace("max_problems = 10", f"max_problems = {max_problems}")
    content = content.replace("rollouts_per_step = 10", f"rollouts_per_step = {max_problems}")
    
    # SRRL needs longer sequences for refined prompts
    if use_srrl:
        content = content.replace("max_length = 512", "max_length = 1024")
    
    # Disable wandb
    content = content.replace('wandb_project = "grpo"', 'wandb_project = None')
    
    # Save modified script
    temp_script = f"train_{method.lower()}_temp.py"
    with open(temp_script, "w") as f:
        f.write(content)
    
    try:
        # Import and run the modified training
        print(f"Starting {method} training...")
        
        # Directly execute the training logic
        import importlib.util
        spec = importlib.util.spec_from_file_location("temp_train", temp_script)
        temp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(temp_module)
        
        # Run training
        temp_module.main()
        
        print(f"✅ {method} training completed!")
        print(f"Model saved to: {output_dir}")
        
        # Clean up
        Path(temp_script).unlink()
        
        return output_dir
        
    except Exception as e:
        print(f"❌ {method} training failed: {e}")
        traceback.print_exc()
        
        # Clean up
        if Path(temp_script).exists():
            Path(temp_script).unlink()
        
        raise e

def run_post_training_evaluation(grpo_path: str, srrl_path: str, test_size: int = 10, num_samples: int = 3) -> Dict[str, Any]:
    """Run comprehensive post-training evaluation."""
    print(f"\n{'='*60}")
    print("POST-TRAINING EVALUATION")
    print(f"{'='*60}")
    
    # Load test problems
    print(f"Loading {test_size} test problems...")
    test_problems = load_mbpp_dataset("test", max_samples=test_size)
    print(f"Loaded {len(test_problems)} test problems")
    
    # Find the actual model paths
    grpo_model_path = Path(grpo_path) / "grpo_final"
    srrl_model_path = Path(srrl_path) / "srrl_final"
    
    if not grpo_model_path.exists():
        # Try alternative naming
        possible_paths = list(Path(grpo_path).glob("*"))
        grpo_model_path = possible_paths[0] if possible_paths else grpo_model_path
    
    if not srrl_model_path.exists():
        # Try alternative naming  
        possible_paths = list(Path(srrl_path).glob("*"))
        srrl_model_path = possible_paths[0] if possible_paths else srrl_model_path
    
    print(f"GRPO model path: {grpo_model_path}")
    print(f"SRRL model path: {srrl_model_path}")
    
    # Run evaluation
    results = compare_models(str(grpo_model_path), str(srrl_model_path), test_problems, num_samples)
    
    print("✅ Post-training evaluation completed!")
    return results

def save_comprehensive_results(training_logs: Dict[str, Any], evaluation_results: Dict[str, Any]) -> str:
    """Save comprehensive results combining training and evaluation data."""
    
    # Build comprehensive results
    comprehensive_results = {
        "experiment_info": {
            "experiment_type": "GRPO vs SRRL Comparison",
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "dataset": "MBPP (Mostly Basic Python Problems)",
            "experiment_date": datetime.now().isoformat(),
            "training_completed": True,
            "evaluation_completed": True
        },
        "training_config": training_logs.get("config", {}),
        "training_results": {
            "grpo": training_logs.get("grpo", {}),
            "srrl": training_logs.get("srrl", {})
        },
        "post_training_evaluation": evaluation_results,
        "final_comparison": {
            "metric": "Pass@k on unseen test problems",
            "grpo_test_accuracy": evaluation_results["grpo"]["pass_at_k"],
            "srrl_test_accuracy": evaluation_results["srrl"]["pass_at_k"], 
            "winner": "GRPO" if evaluation_results["grpo"]["pass_at_k"] > evaluation_results["srrl"]["pass_at_k"] else "SRRL",
            "improvement": evaluation_results["comparison"]["pass_at_k_improvement"]
        },
        "methodology": {
            "training_algorithm": "On-policy GRPO with optional self-refinement",
            "grpo_description": "Base GRPO: c1 completions from original prompt only",
            "srrl_description": "SRRL: c1 + c2 completions (c2 from execution-trace-refined prompts)",
            "evaluation_method": "Pass@k accuracy on unseen MBPP test problems",
            "on_policy_rl": True,
            "weight_updates": True
        }
    }
    
    # Save results
    results_file = "complete_experiment_results.json"
    with open(results_file, "w") as f:
        json.dump(comprehensive_results, f, indent=2)
    
    print(f"📊 Comprehensive results saved to: {results_file}")
    return results_file

def main():
    """Run the complete GRPO vs SRRL experiment."""
    print("🚀 COMPLETE GRPO vs SRRL EXPERIMENT")
    print("="*80)
    print("This will:")
    print("1. Train GRPO model with on-policy RL") 
    print("2. Train SRRL model with execution trace refinement")
    print("3. Evaluate both models on unseen test problems")
    print("4. Generate comprehensive comparison results")
    print("="*80)
    
    training_logs = {
        "config": {
            "max_problems": 6,
            "test_size": 10,
            "num_samples": 3,
            "on_policy": True
        }
    }
    
    try:
        # Phase 1: Train GRPO
        print("\n🔥 PHASE 1: GRPO TRAINING")
        grpo_path = modify_and_run_training("GRPO", "./models/grpo", max_problems=6)
        training_logs["grpo"] = {"status": "completed", "path": grpo_path}
        
        # Phase 2: Train SRRL  
        print("\n🔥 PHASE 2: SRRL TRAINING")
        srrl_path = modify_and_run_training("SRRL", "./models/srrl", max_problems=6)
        training_logs["srrl"] = {"status": "completed", "path": srrl_path}
        
        # Phase 3: Post-training evaluation
        print("\n🎯 PHASE 3: POST-TRAINING EVALUATION")
        evaluation_results = run_post_training_evaluation(grpo_path, srrl_path, test_size=10, num_samples=3)
        
        # Phase 4: Save comprehensive results
        print("\n📊 PHASE 4: SAVING RESULTS")
        results_file = save_comprehensive_results(training_logs, evaluation_results)
        
        # Final summary
        print(f"\n{'='*80}")
        print("🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        grpo_acc = evaluation_results["grpo"]["pass_at_k"]
        srrl_acc = evaluation_results["srrl"]["pass_at_k"]
        
        print(f"📈 FINAL RESULTS (Post-Training Test Accuracy):")
        print(f"   GRPO Pass@3: {grpo_acc:.3f} ({grpo_acc*100:.1f}%)")
        print(f"   SRRL Pass@3: {srrl_acc:.3f} ({srrl_acc*100:.1f}%)")
        
        if srrl_acc > grpo_acc:
            improvement = (srrl_acc - grpo_acc) / grpo_acc * 100 if grpo_acc > 0 else 0
            print(f"🏆 SRRL WINS! (+{improvement:.1f}% improvement)")
        elif grpo_acc > srrl_acc:
            decline = (grpo_acc - srrl_acc) / grpo_acc * 100 if grpo_acc > 0 else 0
            print(f"📊 GRPO WINS! (SRRL is -{decline:.1f}% worse)")
        else:
            print(f"🤝 TIE!")
        
        print(f"\n✅ Training: Both models trained with on-policy RL + weight updates")
        print(f"✅ Evaluation: Both models tested on unseen problems")
        print(f"📄 Full results: {results_file}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ EXPERIMENT FAILED: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 