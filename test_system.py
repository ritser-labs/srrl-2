#!/usr/bin/env python3
"""
Test script to verify the SRRL system works end-to-end.
Run this before the full experiment to catch any issues early.
"""

import torch
import sys
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from mbpp_utils import load_mbpp_dataset, execute_code_with_tests, extract_code_from_completion
        from loss import GRPOLoss
        from replay_buffer import ReplayBuffer, Experience
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_dataset_loading():
    """Test MBPP dataset loading."""
    print("Testing MBPP dataset loading...")
    
    try:
        from mbpp_utils import load_mbpp_dataset
        problems = load_mbpp_dataset("train", max_samples=2)
        
        if len(problems) == 2:
            print("✅ Dataset loading successful")
            print(f"   Sample problem: {problems[0]['task_id']}")
            return True
        else:
            print(f"❌ Expected 2 problems, got {len(problems)}")
            return False
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False


def test_code_execution():
    """Test code execution functionality."""
    print("Testing code execution...")
    
    try:
        from mbpp_utils import execute_code_with_tests
        
        # Test successful code
        code = "def test_func(x):\n    return x * 2"
        tests = ["assert test_func(5) == 10", "assert test_func(0) == 0"]
        result = execute_code_with_tests(code, tests)
        
        if result["success"]:
            print("✅ Code execution successful")
        else:
            print(f"❌ Code execution failed: {result}")
            return False
        
        # Test failing code
        failing_code = "def test_func(x):\n    return x * 3"  # Wrong implementation
        result = execute_code_with_tests(failing_code, tests)
        
        if not result["success"] and len(result["failed_tests"]) > 0:
            print("✅ Code execution failure detection working")
            return True
        else:
            print(f"❌ Failed code should have failed: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Code execution test failed: {e}")
        return False


def test_model_loading():
    """Test model loading (without actually loading the full model)."""
    print("Testing model compatibility...")
    
    try:
        from transformers import AutoTokenizer
        
        # Test tokenizer loading
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
        
        if tokenizer is not None:
            print("✅ Model tokenizer loading successful")
            return True
        else:
            print("❌ Tokenizer loading failed")
            return False
            
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False


def test_refinement_prompt():
    """Test refinement prompt generation."""
    print("Testing refinement prompt generation...")
    
    try:
        from mbpp_utils import create_refinement_prompt
        
        original_prompt = "Write a function to add two numbers"
        failed_code = "def add(a, b):\n    return a - b"  # Wrong implementation
        execution_result = {
            "success": False,
            "error": "",
            "failed_tests": ["assert add(2, 3) == 5: AssertionError"],
            "trace": "FAIL: assert add(2, 3) == 5 - AssertionError"
        }
        
        refined_prompt = create_refinement_prompt(original_prompt, failed_code, execution_result)
        
        if "failed" in refined_prompt.lower() and "add two numbers" in refined_prompt:
            print("✅ Refinement prompt generation successful")
            return True
        else:
            print(f"❌ Refinement prompt seems incorrect: {refined_prompt[:100]}...")
            return False
            
    except Exception as e:
        print(f"❌ Refinement prompt test failed: {e}")
        return False


def test_grpo_loss():
    """Test GRPO loss computation."""
    print("Testing GRPO loss...")
    
    try:
        from loss import GRPOLoss
        from replay_buffer import Experience
        
        # Create dummy experience
        batch_size, seq_len = 2, 10
        experience = Experience(
            sequences=torch.randint(0, 1000, (batch_size, seq_len)),
            action_log_probs=torch.randn(batch_size, seq_len-1),
            log_probs_ref=torch.randn(batch_size, seq_len-1),
            returns=torch.randn(batch_size, 1),
            advantages=torch.randn(batch_size, 1),
            attention_mask=torch.ones(batch_size, seq_len, dtype=torch.bool),
            action_mask=torch.ones(batch_size, seq_len-1, dtype=torch.bool),
        )
        
        loss_fn = GRPOLoss(clip_eps=0.2, kl_weight=0.01)
        log_probs = torch.randn(batch_size, seq_len-1)
        
        loss, kl = loss_fn(log_probs, experience)
        
        if loss.isfinite() and kl.isfinite():
            print("✅ GRPO loss computation successful")
            return True
        else:
            print(f"❌ GRPO loss computation failed: loss={loss}, kl={kl}")
            return False
            
    except Exception as e:
        print(f"❌ GRPO loss test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Running SRRL System Tests")
    print("="*50)
    
    tests = [
        test_imports,
        test_dataset_loading,
        test_code_execution,
        test_model_loading,
        test_refinement_prompt,
        test_grpo_loss,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            print()
    
    print("="*50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for training.")
        print("\nNext steps:")
        print("1. Run: python run_experiment.py --mode both --max_problems 5 --test_size 10")
        print("2. For full experiment: python run_experiment.py --mode both --max_problems 50 --test_size 100")
        return True
    else:
        print("❌ Some tests failed. Please fix issues before running experiments.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 