#!/usr/bin/env python3
"""
Automated experiment runner that trains both GRPO and SRRL models
and generates JSON comparison results automatically.
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

def modify_train_script_for_method(method: str, output_dir: str):
    """Create a training script configured for the specified method."""
    script_name = f"train_{method.lower()}_auto.py"
    
    # Read the original training script
    with open("train_srrl.py", "r") as f:
        content = f.read()
    
    # Set SRRL flag
    use_srrl = method.upper() == "SRRL"
    content = content.replace("use_srrl = True", f"use_srrl = {use_srrl}")
    content = content.replace("use_srrl = False", f"use_srrl = {use_srrl}")
    
    # Set output directory
    content = content.replace('checkpoint_path = Path("./output")', f'checkpoint_path = Path("{output_dir}")')
    
    # Make it a smaller experiment
    content = content.replace("max_problems = 10", "max_problems = 6")
    content = content.replace("rollouts_per_step = 10", "rollouts_per_step = 6")
    
    # Fix max_length for SRRL (refined prompts are longer)
    if method.upper() == "SRRL":
        content = content.replace("max_length = 512", "max_length = 1024")
    
    # Write the modified script
    with open(script_name, "w") as f:
        f.write(content)
    
    return script_name

def train_model(method: str):
    """Train a model using the specified method."""
    print(f"\n{'='*50}")
    print(f"TRAINING {method.upper()} MODEL")
    print(f"{'='*50}")
    
    output_dir = f"./models/{method.lower()}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create training script
    script_name = modify_train_script_for_method(method, output_dir)
    
    try:
        # Run training
        print(f"Starting {method} training...")
        result = subprocess.run([sys.executable, script_name], 
                              check=True, capture_output=True, text=True, timeout=300)
        
        print(f"✅ {method} training completed successfully!")
        print(f"Model saved to: {output_dir}")
        
        # Show last few lines of output for progress
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            print("Training progress:")
            for line in lines[-10:]:  # Last 10 lines
                if "Step" in line or "Epoch" in line:
                    print(f"  {line}")
        
        # Clean up temporary script
        Path(script_name).unlink()
        
        return output_dir
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {method} training failed!")
        print(f"Error: {e.stderr}")
        
        # Clean up temporary script
        if Path(script_name).exists():
            Path(script_name).unlink()
            
        raise e
    except subprocess.TimeoutExpired:
        print(f"❌ {method} training timed out!")
        
        # Clean up temporary script
        if Path(script_name).exists():
            Path(script_name).unlink()
            
        raise

def run_evaluation(grpo_path: str, srrl_path: str):
    """Run evaluation and generate JSON results."""
    print(f"\n{'='*50}")
    print("RUNNING EVALUATION")
    print(f"{'='*50}")
    
    cmd = [
        sys.executable, "evaluate.py",
        "--grpo_model", grpo_path,
        "--srrl_model", srrl_path,
        "--test_size", "10",
        "--num_samples", "3"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=180)
        print("✅ Evaluation completed successfully!")
        
        # Show the output
        if result.stdout:
            print(result.stdout)
        
        # Check if results file was created
        results_file = Path("evaluation_results.json")
        if results_file.exists():
            print(f"📊 Results saved to: {results_file}")
            return results_file
        else:
            print("⚠️ Results file not found")
            return None
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed!")
        print(f"Error: {e.stderr}")
        raise e
    except subprocess.TimeoutExpired:
        print(f"❌ Evaluation timed out!")
        raise

def main():
    """Run complete automated experiment."""
    print("🚀 Automated GRPO vs SRRL Experiment")
    print("="*60)
    
    try:
        # Train GRPO
        grpo_path = train_model("GRPO")
        
        # Train SRRL  
        srrl_path = train_model("SRRL")
        
        # Run evaluation
        results_file = run_evaluation(grpo_path, srrl_path)
        
        if results_file and results_file.exists():
            # Load and display results
            with open(results_file, "r") as f:
                results = json.load(f)
            
            print(f"\n{'='*60}")
            print("🎉 EXPERIMENT COMPLETED!")
            print("="*60)
            
            grpo_score = results['grpo']['pass_at_k']
            srrl_score = results['srrl']['pass_at_k']
            
            print(f"GRPO Pass@3: {grpo_score:.3f}")
            print(f"SRRL Pass@3: {srrl_score:.3f}")
            
            if srrl_score > grpo_score:
                improvement = (srrl_score - grpo_score) / grpo_score * 100 if grpo_score > 0 else 0
                print(f"🏆 SRRL WINS! (+{improvement:.1f}% improvement)")
            elif grpo_score > srrl_score:
                decline = (grpo_score - srrl_score) / grpo_score * 100 if grpo_score > 0 else 0
                print(f"📊 GRPO WINS! (SRRL is -{decline:.1f}% worse)")
            else:
                print(f"🤝 TIE!")
                
            print(f"📄 Detailed results: {results_file}")
            print("\n✅ Automated experiment completed successfully!")
        else:
            print("❌ Could not generate results file")
            return 1
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 