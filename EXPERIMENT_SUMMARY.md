# SRRL vs GRPO Experiment Implementation Summary

## What We Built

This repository implements a complete experimental framework comparing **GRPO (Group Relative Policy Optimization)** with **SRRL (Self-Refined RL)** for code generation tasks.

## Key Innovation: SRRL Method

**SRRL** introduces a novel refinement mechanism to reinforcement learning for code generation:

### Traditional GRPO Flow:
```
Original Prompt → Sample Completions (c1) → Execute & Calculate Rewards → Train
```

### SRRL Flow:
```
Original Prompt → Sample Completions (c1) → Execute & Identify Failures
                                                      ↓
                                            Generate Execution Traces
                                                      ↓
                                            Create Refined Prompts
                                                      ↓
                           Train ← Sample Completions (c2) ← Refined Prompts
```

## Technical Implementation

### Core Components

1. **`mbpp_utils.py`** - Dataset & execution infrastructure
   - Safe code execution with test cases
   - MBPP dataset loading and processing
   - Execution trace generation for failed attempts
   - Code extraction from model completions

2. **`train_srrl.py`** - Main training script
   - Supports both GRPO and SRRL modes
   - Implements the two-phase rollout system (c1 + c2)
   - Balances compute between original and refined attempts
   - Uses Qwen/Qwen2.5-3B-Instruct model

3. **`evaluate.py`** - Comprehensive evaluation
   - Pass@1 and Pass@k metrics
   - Problem-by-problem analysis
   - Detailed comparison statistics

4. **`run_experiment.py`** - Experiment orchestration
   - Automated training and evaluation pipeline
   - Configurable hyperparameters
   - Easy switching between methods

5. **`test_system.py`** - System validation
   - End-to-end testing before experiments
   - Catches integration issues early

6. **`visualize_results.py`** - Result analysis
   - Comparative performance plots
   - Improvement breakdown analysis
   - Problem-wise success distribution

### Key Features

- **Safe Execution**: Sandboxed code execution environment
- **Detailed Tracing**: Comprehensive error messages and execution traces
- **Compute Control**: Balanced c1/c2 sampling for fair comparison
- **High Observability**: Extensive logging and metrics
- **Clean Architecture**: Modular design minimizing bugs

## SRRL's Refinement Process

When SRRL encounters a failed code attempt, it:

1. **Captures Execution State**: Records the exact error and trace
2. **Identifies Root Cause**: Determines which test cases failed and why
3. **Creates Refined Prompt**: 
   ```
   The following code attempt failed to solve the problem correctly:
   
   Original Problem: [original prompt]
   
   Failed Code: [generated code]
   
   Execution Result:
   Error: [specific error message]
   Failed Tests: [which tests failed and how]
   Execution Trace: [detailed execution trace]
   
   Please analyze the error and provide a corrected solution...
   ```
4. **Samples Corrections**: Generates new attempts from refined prompt
5. **Trains on Both**: Uses both original (c1) and refined (c2) experiences

## Expected Benefits

SRRL should outperform GRPO because it provides:

1. **Explicit Error Feedback**: Models learn from specific failure modes
2. **Iterative Refinement**: Opportunity to fix mistakes with guidance
3. **Richer Training Data**: Both initial attempts and corrected versions
4. **Better Generalization**: Exposure to error-correction patterns

## Usage Examples

### Quick Test Run:
```bash
python run_experiment.py --mode both --max_problems 5 --test_size 10
```

### Full Experiment:
```bash
python run_experiment.py --mode both --max_problems 50 --test_size 100
```

### Results Analysis:
```bash
python visualize_results.py
```

## File Structure

```
├── mbpp_utils.py           # Dataset & execution utilities
├── train_srrl.py          # Main training (GRPO + SRRL)
├── evaluate.py            # Model evaluation & comparison
├── run_experiment.py      # Experiment orchestration
├── test_system.py         # System validation
├── visualize_results.py   # Result visualization
├── loss.py               # GRPO loss implementation
├── replay_buffer.py      # Experience replay buffer
└── README.md             # Usage documentation
```

## Experiment Design

### Controlled Comparison
- **Same Model**: Qwen/Qwen2.5-3B-Instruct for both methods
- **Same Dataset**: MBPP coding problems with test cases
- **Same Hyperparameters**: Learning rate, batch size, etc.
- **Same Evaluation**: Pass@k metrics on held-out test set

### Compute Balancing
- SRRL uses c1 + limited c2 samples
- Total compute controlled to ensure fair comparison
- Configurable c1/c2 ratio for ablation studies

## Innovation Summary

This implementation represents a novel approach to RL for code generation by:

1. **Introducing Error-Aware Training**: Using execution feedback to guide learning
2. **Implementing Self-Refinement**: Allowing models to correct their mistakes
3. **Maintaining Compute Efficiency**: Balanced sampling strategy
4. **Providing Comprehensive Evaluation**: Detailed comparison framework

The SRRL method bridges the gap between traditional RL approaches and more sophisticated error-correction mechanisms, potentially leading to better code generation capabilities through iterative refinement.

## Next Steps

After running the experiment, researchers can:

1. **Analyze Results**: Use visualization tools to understand improvements
2. **Ablation Studies**: Vary c1/c2 ratios, refinement strategies
3. **Scale Up**: Apply to larger models and datasets
4. **Extend Methods**: Combine with other RL improvements 