#!/usr/bin/env python3
"""
Simple experiment runner that directly executes GRPO and SRRL training
and generates JSON comparison results automatically.
"""

import os
import sys
import json
import torch
from pathlib import Path

# Import our training and evaluation modules
sys.path.append('.')

def run_grpo_training():
    """Run GRPO training directly."""
    print("🔥 Training GRPO model...")
    
    # Import and modify the training configuration for GRPO
    import train_srrl
    
    # Backup original main function
    original_main = train_srrl.main
    
    # Create a GRPO-specific configuration
    def grpo_main():
        # Configuration for GRPO
        use_srrl = False  # GRPO mode
        seed = 42
        wandb_project = None
        device_index = 0
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        checkpoint_path = Path("./models/grpo")
        checkpoint_interval = 20
        train_batch_size = 4
        lr = 1e-5
        kl_weight = 0.01
        clip_eps = 0.2
        group_size = 4
        rollouts_per_step = 8  # Small experiment
        epochs_per_step = 1
        max_norm = 1.0
        max_length = 512
        top_p = 0.9
        temperature = 0.8
        max_problems = 6  # Small experiment
        
        # Make checkpoint directory
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Run the training logic directly
        from mbpp_utils import load_mbpp_dataset
        from replay_buffer import ReplayBuffer, Experience
        from torch.utils.data import DataLoader
        
        device = torch.device("cuda", device_index)
        torch.manual_seed(seed)
        
        # Load model
        model, tokenizer = train_srrl.load_model(model_name, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
        # Load data
        problems = load_mbpp_dataset("train", max_samples=max_problems)
        dataloader = DataLoader(problems, batch_size=1, shuffle=True)
        
        replay_buffer = ReplayBuffer(capacity=1000)
        
        print(f"Loaded {len(problems)} MBPP problems")
        print("Starting GRPO training...")
        
        for step, problem_batch in enumerate(dataloader):
            if step >= rollouts_per_step:
                break
                
            problem = {
                'task_id': problem_batch['task_id'].item(),
                'prompt': problem_batch['prompt'][0],
                'code': problem_batch['code'][0],
                'test_list': problem_batch['test_list'][0],
                'test_imports': problem_batch['test_imports'][0] if 'test_imports' in problem_batch and len(problem_batch['test_imports']) > 0 and len(problem_batch['test_imports'][0]) > 0 else []
            }
            
            # Generate completions and rewards (GRPO style - no refinement)
            experiences = train_srrl.rollout_with_rewards(
                model, tokenizer, problem, group_size, 
                max_length, top_p, temperature, use_refinement=use_srrl, device=device
            )
            
            for exp in experiences:
                replay_buffer.add(exp)
            
            # Calculate metrics
            returns = sum(exp.reward for exp in experiences) 
            success_rate = sum(1 for exp in experiences if exp.reward >= 1.0) / len(experiences)
            
            print(f"Step {step}: returns={returns:.4f}, success_rate={success_rate:.4f}")
            
            # Training step
            if len(replay_buffer) >= train_batch_size:
                batch = replay_buffer.sample(train_batch_size)
                loss, kl, grad_norm = train_srrl.training_step(model, optimizer, batch, kl_weight, clip_eps, max_norm)
                print(f"Epoch 0: loss={loss:.4f}, kl={kl:.4f}, grad_norm={grad_norm:.4f}")
        
        # Save model
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        print(f"✅ GRPO model saved to {checkpoint_path}")
        
        return checkpoint_path
    
    # Run GRPO training
    return grpo_main()

def run_srrl_training():
    """Run SRRL training directly."""
    print("🔥 Training SRRL model...")
    
    # Import and modify the training configuration for SRRL
    import train_srrl
    
    def srrl_main():
        # Configuration for SRRL
        use_srrl = True  # SRRL mode with refinement
        seed = 42
        wandb_project = None
        device_index = 0
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        checkpoint_path = Path("./models/srrl")
        checkpoint_interval = 20
        train_batch_size = 4
        lr = 1e-5
        kl_weight = 0.01
        clip_eps = 0.2
        group_size = 4
        rollouts_per_step = 8  # Small experiment
        epochs_per_step = 1
        max_norm = 1.0
        max_length = 512
        top_p = 0.9
        temperature = 0.8
        max_problems = 6  # Small experiment
        
        # Make checkpoint directory
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Run the training logic directly
        from mbpp_utils import load_mbpp_dataset
        from replay_buffer import ReplayBuffer, Experience
        from torch.utils.data import DataLoader
        
        device = torch.device("cuda", device_index)
        torch.manual_seed(seed)
        
        # Load model
        model, tokenizer = train_srrl.load_model(model_name, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
        # Load data
        problems = load_mbpp_dataset("train", max_samples=max_problems)
        dataloader = DataLoader(problems, batch_size=1, shuffle=True)
        
        replay_buffer = ReplayBuffer(capacity=1000)
        
        print(f"Loaded {len(problems)} MBPP problems")
        print("Starting SRRL training...")
        
        for step, problem_batch in enumerate(dataloader):
            if step >= rollouts_per_step:
                break
                
            problem = {
                'task_id': problem_batch['task_id'].item(),
                'prompt': problem_batch['prompt'][0],
                'code': problem_batch['code'][0],
                'test_list': problem_batch['test_list'][0],
                'test_imports': problem_batch['test_imports'][0] if 'test_imports' in problem_batch and len(problem_batch['test_imports']) > 0 and len(problem_batch['test_imports'][0]) > 0 else []
            }
            
            # Generate completions and rewards (SRRL style - with refinement)
            experiences = train_srrl.rollout_with_rewards(
                model, tokenizer, problem, group_size, 
                max_length, top_p, temperature, use_refinement=use_srrl, device=device
            )
            
            for exp in experiences:
                replay_buffer.add(exp)
            
            # Calculate metrics
            returns = sum(exp.reward for exp in experiences) 
            success_rate = sum(1 for exp in experiences if exp.reward >= 1.0) / len(experiences)
            
            print(f"Step {step}: returns={returns:.4f}, success_rate={success_rate:.4f}")
            
            # Training step
            if len(replay_buffer) >= train_batch_size:
                batch = replay_buffer.sample(train_batch_size)
                loss, kl, grad_norm = train_srrl.training_step(model, optimizer, batch, kl_weight, clip_eps, max_norm)
                print(f"Epoch 0: loss={loss:.4f}, kl={kl:.4f}, grad_norm={grad_norm:.4f}")
        
        # Save model
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        print(f"✅ SRRL model saved to {checkpoint_path}")
        
        return checkpoint_path
    
    # Run SRRL training
    return srrl_main()

def run_evaluation(grpo_path, srrl_path):
    """Run evaluation and generate JSON results."""
    print("🔬 Running evaluation...")
    
    # Import evaluation functions
    from evaluate import evaluate_model, compare_models
    from mbpp_utils import load_mbpp_dataset
    
    # Load test problems
    test_problems = load_mbpp_dataset("test", max_samples=10)
    print(f"Loaded {len(test_problems)} test problems")
    
    # Run comparison
    results = compare_models(str(grpo_path), str(srrl_path), test_problems, num_samples=3)
    
    print("✅ Evaluation completed and results saved to evaluation_results.json")
    return results

def main():
    """Run complete experiment."""
    print("🚀 Starting Automated GRPO vs SRRL Experiment")
    print("="*60)
    
    try:
        # Train GRPO
        print("\n" + "="*30)
        print("PHASE 1: GRPO TRAINING") 
        print("="*30)
        grpo_path = run_grpo_training()
        
        # Train SRRL
        print("\n" + "="*30)
        print("PHASE 2: SRRL TRAINING")
        print("="*30)
        srrl_path = run_srrl_training()
        
        # Run evaluation
        print("\n" + "="*30)
        print("PHASE 3: EVALUATION")
        print("="*30)
        results = run_evaluation(grpo_path, srrl_path)
        
        # Print final summary  
        print("\n" + "="*60)
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
            decline = (grpo_score - srrl_score) / grpo_score * 100
            print(f"📊 GRPO WINS! (SRRL is -{decline:.1f}% worse)")
        else:
            print(f"🤝 TIE!")
            
        print(f"📄 Detailed results saved to: evaluation_results.json")
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 