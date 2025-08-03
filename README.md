# SRRL vs GRPO Code Generation Experiment

**Fork of the original GRPO implementation** - This repository implements and compares two reinforcement learning methods for code generation on the MBPP dataset.

## Methods Compared

### 1. GRPO (Group Relative Policy Optimization) - Baseline
- Samples multiple code completions from problem prompts
- Executes code against test cases to compute rewards
- Trains policy using group-relative advantages

### 2. SRRL (Self-Refined Reinforcement Learning) - Novel Method
- Basically GRPO but with a self-reflection step.
- **Phase 1 (c1)**: Generate initial code attempts from original prompts
- **Phase 2 (c2)**: For failed attempts, create refined prompts with:
  - Detailed execution traces and error messages
  - Concrete input/output examples showing expected vs actual results
  - Self-reflection analysis sections for better debugging
- **Training**: Policy gradient updates on both c1 and c2 completions with per-problem advantage normalization

## Key Experiment Features

### Dataset & Scale
- **Dataset**: MBPP (Mostly Basic Python Problems) - 30 training problems, 30 test problems
- **Model**: Qwen/Qwen2.5-3B-Instruct with Flash-Attention 2 (bfloat16)
- **Compute Control**: GRPO uses 10 completions, SRRL uses 2 c1 + up to 8 c2 completions

### Advanced Execution & Feedback
- **Sandboxed Execution**: Process-isolated code execution with hard timeouts (10s)
- **Rich Error Feedback**: Shows actual vs expected values for failed assertions
- **Timeout Handling**: Detects infinite loops and provides specific timeout feedback
- **Test Result Parsing**: Detailed PASS/FAIL logs with concrete return values

### Training Improvements
- **Per-Problem Training**: Advantage normalization per problem for stronger c2 signals
- **Multi-Sample c2**: 4 refinement attempts per failure with temperature=0.3 for focused exploration
- **Flash-Attention**: Memory-efficient attention for longer sequences
- **Comprehensive Logging**: W&B integration tracking every completion and reward

## Quick Start

### Run Full Experiment
```bash
# Train both methods and evaluate
python experiment.py

# Train only GRPO
python experiment.py --method grpo

# Train only SRRL  
python experiment.py --method srrl
```

### Key Files
- `experiment.py` - Main experiment runner with W&B logging
- `mbpp_utils.py` - Dataset loading and sandboxed code execution
- `loss.py` - GRPO loss implementation

## Expected Results

SRRL aims to outperform GRPO through:

1. **Concrete Error Feedback**: Shows exact input/output mismatches rather than generic error messages
2. **Self-Reflection**: Allows model to analyze failures before attempting fixes
3. **Targeted Exploration**: Lower temperature c2 generation focuses on bug fixes rather than random exploration
4. **Stronger Training Signal**: Per-problem advantage normalization gives successful refinements higher weight

The experiment tracks:
- **c2 Success Rate**: Percentage of failed c1 attempts rescued by refinement
- **Overall Accuracy**: Pass@1 on held-out test problems
- **Training Dynamics**: Reward distributions and convergence patterns

## Method Details

### Refinement Prompt Structure
```
The following code attempt failed to solve the problem correctly:

Original Problem: [problem description]

Failed Code:
```python
[generated code]
```

Execution Result:
FAIL: assert func("input") == "expected" -> got "actual" (expected "expected")
Error: [detailed traceback]
Execution Trace: [stdout/stderr]

Feel free to reason in an <analysis> section. Afterwards output the fixed code inside <code>...</code>

Format:
<analysis>
...thoughts...
</analysis>

<code>
def func(...):
    ...
</code>
```

### Training Process
1. **GRPO**: Standard policy gradient on 10 completions per problem
2. **SRRL**: 
   - Generate 2 c1 completions
   - For each failure, generate 4 c2 refinement attempts
   - Train on all completions with per-problem advantage baseline
   - Track c2 rescue rate and refinement statistics

## Monitoring & Results

The experiment logs to Weights & Biases:
- Individual completion rewards (phase: grpo_c1, srrl_c1, srrl_c2)
- Per-problem c2 success rates and refinement counts
- Final accuracy comparison between methods

Console output shows:
- Full code completions during training (for debugging)
- Detailed execution traces for failures
- c2 success rate per problem
- Final accuracy comparison

## References

- [MBPP Dataset](https://github.com/google-research/google-research/tree/master/mbpp)
- [Qwen2.5 Models](https://huggingface.co/Qwen)  
- [Flash-Attention](https://github.com/Dao-AILab/flash-attention)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)