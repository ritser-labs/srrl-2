# GRPO vs SRRL Experiment

This repository implements and compares two reinforcement learning methods for code generation:

1. **GRPO (Group Relative Policy Optimization)** - The baseline method
2. **SRRL (Self-Refined RL)** - A novel method that uses execution feedback to refine failed attempts

## Overview

**GRPO** samples completions from prompts and trains on the rewards based on test case execution.

**SRRL** extends GRPO by:
1. Sampling initial completions (c1) from the original prompt
2. Identifying failed completions and generating execution traces
3. Creating refined prompts with error feedback for failed attempts
4. Sampling refined completions (c2) from the refined prompts  
5. Training on both c1 and c2 completions together

## Dataset & Model

- **Dataset**: MBPP (Mostly Basic Python Problems) - coding problems with test cases
- **Model**: Qwen/Qwen2.5-3B-Instruct - instruction-tuned model for code generation
- **Evaluation**: Pass@k accuracy on held-out test set

## Quick Start

### 1. Setup Environment

```bash
conda create --name srrl python=3.12 -y
conda activate srrl
pip install -r requirements.txt
```

### 2. Run Full Experiment (Train + Evaluate)

```bash
python run_experiment.py --mode both --max_problems 50 --test_size 100
```

This will:
1. Train a GRPO model 
2. Train an SRRL model
3. Evaluate both models on test set
4. Generate comparison results

### 3. Run Training Only

```bash
# Train GRPO baseline
python run_experiment.py --mode train --max_problems 50

# Or train specific method by modifying train_srrl.py:
# Set use_srrl = False for GRPO
# Set use_srrl = True for SRRL
python train_srrl.py
```

### 4. Run Evaluation Only

```bash
python run_experiment.py --mode eval \
    --grpo_path ./output/grpo_final \
    --srrl_path ./output/srrl_final \
    --test_size 100 --num_samples 5
```

### 5. Manual Evaluation

```bash
python evaluate.py \
    --grpo_model ./output/grpo_final \
    --srrl_model ./output/srrl_final \
    --test_size 100 --num_samples 5
```

## Key Components

### Core Files

- `train_srrl.py` - Main training script supporting both GRPO and SRRL
- `mbpp_utils.py` - Dataset loading and code execution utilities
- `evaluate.py` - Model evaluation and comparison script
- `run_experiment.py` - Experiment runner for easy GRPO vs SRRL comparison
- `loss.py` - GRPO loss implementation
- `replay_buffer.py` - Experience replay buffer for RL training

### Key Features

- **Safe Code Execution**: Sandboxed execution environment for test cases
- **Execution Traces**: Detailed error feedback for failed code attempts  
- **Compute Control**: Balanced c1/c2 sampling to control computational overhead
- **Comprehensive Metrics**: Pass@1, Pass@k, average rewards, success rates
- **Experiment Tracking**: WandB integration for training monitoring

## Configuration

Key hyperparameters in `train_srrl.py`:

```python
use_srrl = True              # Enable/disable SRRL refinement
max_problems = 50            # Training dataset size  
group_size = 8               # Completions per problem
rollouts_per_step = 16       # Problems per training step
lr = 1e-5                    # Learning rate
temperature = 0.8            # Sampling temperature
max_length = 1024            # Max sequence length
```

## Expected Results

SRRL should outperform GRPO by providing the model with:
1. **Error feedback** - Explicit information about why code failed
2. **Refinement opportunities** - Chance to fix mistakes with guided prompts
3. **Diverse experiences** - Both original attempts and refined attempts for training

The evaluation script will show improvements in Pass@k accuracy and average rewards.

## Method Details

### GRPO Baseline
1. Sample completions from original prompts
2. Execute code with test cases to get rewards
3. Train with GRPO loss on policy optimization

### SRRL Method  
1. Sample c1 completions from original prompts
2. Execute and identify failed attempts
3. Generate refined prompts with execution traces
4. Sample c2 completions from refined prompts  
5. Train with GRPO loss on combined c1+c2 experiences

### Refinement Prompt Template
```
The following code attempt failed to solve the problem correctly:

Original Problem: [problem description]

Failed Code: [generated code]

Execution Result:
Error: [error message]
Failed Tests: [specific test failures]
Execution Trace: [detailed trace]

Please analyze the error and provide a corrected solution...
```

## Contributing

To add new features or methods:
1. Extend `mbpp_utils.py` for dataset/execution utilities
2. Modify `train_srrl.py` for new training procedures  
3. Update `evaluate.py` for new evaluation metrics
4. Add configuration options to `run_experiment.py`

## References

- [MBPP Dataset](https://github.com/google-research/google-research/tree/master/mbpp)  
- [Qwen2.5 Models](https://huggingface.co/Qwen)
- [GRPO: Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
