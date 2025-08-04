#!/usr/bin/env python3
"""
NUMERICALLY STABLE GRPO vs SRRL Experiment
"""

import os
import json
import torch
import re
import math
import argparse
import wandb
from pathlib import Path
from datetime import datetime
import functools, builtins  # Added for auto-flushing prints
print = functools.partial(builtins.print, flush=True)  # Auto-flush all prints
MAX_TRAIN_TOKENS = 1024  # Limit sequence length during training to avoid OOM

# ---------- Compute budget (keep methods comparable) ----------
MAX_COMPLETION_BUDGET = 10     # Unified number of completions (per-problem) allowed
GROUP_SIZE_GRPO = MAX_COMPLETION_BUDGET  # GRPO consumes the full budget up-front
GROUP_SIZE_SRRL = 2            # SRRL initial (c1) completions; remaining budget used for refinements
from loss import GRPOLoss, approx_kl_divergence

# --------------------- W&B Logging Helpers ---------------------

def log_completion(table_name: str, **kwargs):
    """Utility to accumulate rows in a W&B Table.

    Stores a set of lightweight W&B Tables on the active run so we only
    need to call `wandb.log` once per table, avoiding huge numbers of
    individual log calls that slow the run down.
    """
    global _WB_TABLE_CACHE
    if "_WB_TABLE_CACHE" not in globals():
        _WB_TABLE_CACHE = {}
    if table_name not in _WB_TABLE_CACHE:
        _WB_TABLE_CACHE[table_name] = wandb.Table(columns=list(kwargs.keys()))
    _WB_TABLE_CACHE[table_name].add_data(*kwargs.values())

def flush_logged_tables():
    """Flush all cached tables to W&B so they are persisted."""
    global _WB_TABLE_CACHE
    for name, tbl in globals().get("_WB_TABLE_CACHE", {}).items():
        wandb.log({name: tbl})

def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # `torch.cuda.synchronize()` can hang if the device is in an error state.
        if os.getenv("CUDA_SYNC", "0") == "1":
            try:
                torch.cuda.synchronize()
            except RuntimeError as e:
                print(f"⚠️ cuda.synchronize skipped due to error: {e}")

def cleanup_model(model):
    """Properly cleanup model to prevent CUDA corruption."""
    if model is not None:
        model.cpu()
        del model
        clear_memory()

def create_code_prompt(problem):
    """Create a clear prompt that instructs LLM to generate Python functions."""
    first_test = problem["test_list"][0] if problem["test_list"] else ""
    function_name = "unknown_function"
    
    match = re.search(r'(\w+)\s*\(', first_test)
    if match:
        function_name = match.group(1)

    prompt = f"""Problem: {problem['prompt']}

    Test: {first_test}

    You may first use an <analysis> section to think step-by-step. Anything inside <analysis> will be hidden from evaluation.

    Then output the solution inside a <code> block.

    Format exactly:

    <analysis>
    ... your reasoning ...
    </analysis>

    <code>
    def {function_name}(
        # your implementation
    </code>"""
    
    return prompt

def extract_code_from_completion(completion):
    """Extract Python code from XML-formatted completion."""
    code_match = re.search(r'<code>\s*(.*?)\s*</code>', completion, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    
    if "```python" in completion:
        start = completion.find("```python") + 9
        end = completion.find("```", start)
        if end != -1:
            return completion[start:end].strip()
    
    if "```" in completion:
        start = completion.find("```") + 3
        end = completion.find("```", start)
        if end != -1:
            return completion[start:end].strip()
    
    lines = completion.split('\n')
    code_lines = []
    in_function = False
    
    for line in lines:
        if line.strip().startswith('def '):
            in_function = True
            code_lines.append(line)
        elif in_function:
            if line.strip() == "" or line.startswith('    ') or line.startswith('\t'):
                code_lines.append(line)
            else:
                break
    
    if code_lines:
        return '\n'.join(code_lines)
    
    return completion.strip()

def create_refinement_prompt(original_prompt, failed_code, execution_result):
    """Create refinement prompt with execution feedback."""
    function_name = "unknown_function"
    if execution_result.get("failed_tests"):
        first_test = execution_result["failed_tests"][0]
        match = re.search(r'(\w+)\s*\(', first_test)
        if match:
            function_name = match.group(1)
    
    refinement_prompt = f"""The following code attempt failed to solve the problem correctly:

Original Problem:
{original_prompt}

Failed Code:
```python
{failed_code}
```

Execution Result:
"""
    
    if execution_result.get("error"):
        refinement_prompt += f"Error: {execution_result['error']}\n"
    
    if execution_result.get('failed_tests'):
        refinement_prompt += "Failed Tests:\n"
        for test in execution_result['failed_tests']:
            refinement_prompt += f"- {test}\n"

    if execution_result.get('trace'):
        refinement_prompt += f"\nExecution Trace:\n{execution_result['trace']}\n"

    refinement_prompt += f"""
Feel free to reason in an <analysis> section. Afterwards output the fixed code inside <code>...</code> as before.

Format:
<analysis>
...thoughts...
</analysis>

<code>
def {function_name}(...):
    ...
</code>"""
    
    return refinement_prompt

def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize advantages like the original GRPO code."""
    return (returns - returns.mean()) / (returns.std() + eps)

def sequence_log_probs_from_logits(logits: torch.tensor, output_ids: torch.tensor) -> torch.Tensor:
    """Extract log probs like the original code."""
    log_prob = torch.nn.functional.log_softmax(logits, dim=-1)
    return log_prob.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)

def sequences_log_probs(model, sequence_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Get sequence log probs like the original code."""
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
    output = model.forward(
        input_ids=sequence_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    logits = output["logits"]
    log_probs = sequence_log_probs_from_logits(
        logits=logits[:, :-1].to(torch.float32),
        output_ids=sequence_ids[:, 1:],
    )
    return log_probs

# ===============================================
# Helper: sample-wise training to avoid OOM
def stable_train_sample_wise(model, optimizer, tokenizer, sequences, rewards, device, label="GRPO"):
    """Perform a single policy-gradient update sample-by-sample.
    This keeps VRAM usage low and avoids mixing problems together."""
    if not sequences:
        return
    model.train()
    optimizer.zero_grad()
    rewards_tensor = torch.tensor(rewards, device=device, dtype=torch.float32)
    advantages = group_advantages(rewards_tensor).tolist()
    for seq_idx, (seq, adv) in enumerate(zip(sequences, advantages)):
        seq = seq[-MAX_TRAIN_TOKENS:]
        sequence_ids = torch.cat([
            torch.full((MAX_TRAIN_TOKENS - len(seq),), tokenizer.pad_token_id, device=device),
            seq
        ]).unsqueeze(0)
        attention_mask = (sequence_ids != tokenizer.pad_token_id)
        log_probs = sequences_log_probs(model, sequence_ids, attention_mask)
        attn_shifted = attention_mask[:, 1:]
        avg_log_prob = (log_probs * attn_shifted).sum() / attn_shifted.sum().clamp(min=1)
        loss = -(avg_log_prob * adv)
        loss.backward()
        
        if (seq_idx + 1) % 1 == 0 or seq_idx == len(sequences) - 1:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if math.isfinite(grad_norm):
                optimizer.step()
                optimizer.zero_grad()
                with torch.no_grad():
                    for p in model.parameters():
                        if torch.isnan(p).any() or torch.isinf(p).any():
                            raise RuntimeError("Detected NaN/Inf in model parameters after update")
    print(f"      ✅ {label} sample-wise update done ({len(sequences)} sequences)")
    clear_memory()
# ===============================================

def train_grpo(train_problems, num_steps=1):  # Reduce to 1 step to avoid corruption
    print("\n🚀 TRAINING GRPO (PROPER LOSS)")
    print("=" * 40)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from mbpp_utils import execute_code_with_tests, calculate_code_reward
        
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2")
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-8, weight_decay=0.01)
        
        print(f"Training on {len(train_problems)} problems")
        
        for step in range(num_steps):
            print(f"Step {step+1}/{num_steps}")
            comp_counter = 0
            
            total_rewards = []
            total_sequences = []
            
            for idx, problem in enumerate(train_problems):
                print(f"\n  ⚙️ GRPO Problem {idx+1}/{len(train_problems)}: {problem['task_id']}")
                prompt = create_code_prompt(problem)
                
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                
                model.eval()
                with torch.no_grad():
                    try:
                        outputs = model.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=1024,  # unified token budget
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            num_return_sequences=GROUP_SIZE_GRPO,
                            pad_token_id=tokenizer.pad_token_id,
                            use_cache=True
                        )
                    except RuntimeError as e:
                        if "CUDA" in str(e):
                            print(f"    ⚠️ CUDA error during generation, skipping problem {problem['task_id']}")
                            continue
                        else:
                            raise
                
                local_rewards = []
                local_sequences = []
                
                for output in outputs:
                    completion = tokenizer.decode(output[input_ids.shape[1]:], skip_special_tokens=True)
                    code = extract_code_from_completion(completion)
                    
                    exec_result = execute_code_with_tests(code, problem["test_list"], problem.get("test_imports", []))
                    reward = calculate_code_reward(exec_result)
                    local_rewards.append(reward)
                    wandb.log({"phase":"grpo_c1","task_id": problem['task_id'],"reward": reward})
                    log_completion("grpo_c1_table", phase="grpo_c1", step=step+1, task_id=problem['task_id'], completion_idx=comp_counter, reward=reward, success=exec_result["success"], code=code)
                    comp_counter += 1
                    # encode only the code tokens for training
                    code_ids = tokenizer(code, return_tensors="pt", truncation=True, max_length=MAX_TRAIN_TOKENS).input_ids[0].to(device)
                    local_sequences.append(code_ids)
                    
                    total_rewards.append(reward)
                    total_sequences.append(code_ids)
                    
                    if reward >= 1.0:
                        print(f"    ✅ Success! Reward: {reward:.2f}")
                    else:
                        print(f"    ❌ Failed. Reward: {reward:.2f}")
                
                # --- train per-problem ---
                if any(r > 0 for r in local_rewards):
                    stable_train_sample_wise(model, optimizer, tokenizer, local_sequences, local_rewards, device, label="GRPO")
            # end for problem loop
            
            if not total_rewards:
                print(f"  No valid sequences, skipping training step")
                continue
                
            avg_reward = sum(total_rewards) / len(total_rewards)
            success_rate = sum(1 for r in total_rewards if r >= 1.0) / len(total_rewards)
            print(f"  Avg reward: {avg_reward:.3f}, Success: {success_rate:.1%}")
            
            # Clear model state and memory to prevent corruption
            model.zero_grad(set_to_none=True)
            clear_memory()
        
        output_dir = "./models/grpo"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print("✅ GRPO completed!")
        
        # Cleanup model properly to prevent CUDA corruption
        cleanup_model(model)
        return True
        
    except Exception as e:
        print(f"❌ GRPO failed: {e}")
        import traceback
        traceback.print_exc()
        if 'model' in locals():
            cleanup_model(model)
        return False

def train_srrl(train_problems, num_steps=1):  # Reduce to 1 step to avoid corruption
    print("\n🚀 TRAINING SRRL (PROPER LOSS)")
    print("=" * 40)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from mbpp_utils import execute_code_with_tests, calculate_code_reward
        
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2")
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-8, weight_decay=0.01)
        
        print(f"Training on {len(train_problems)} problems")
        
        for step in range(num_steps):
            print(f"Step {step+1}/{num_steps}")
            comp_counter = 0
            
            all_rewards = []
            all_sequences = []
            refinement_count = 0
            
            for idx, problem in enumerate(train_problems):
                print(f"\n  ⚙️ SRRL Problem {idx+1}/{len(train_problems)}: {problem['task_id']}")
                prompt = create_code_prompt(problem)
                
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                
                model.eval()
                failed_attempts = []
                local_rewards_sr = []
                local_sequences_sr = []
                c2_successes = 0
                
                with torch.no_grad():
                    try:
                        outputs = model.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=1024,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            num_return_sequences=GROUP_SIZE_SRRL,
                            pad_token_id=tokenizer.pad_token_id,
                            use_cache=True
                        )
                    except RuntimeError as e:
                        if "CUDA" in str(e):
                            print(f"    ⚠️ CUDA error during c1 generation, skipping")
                            continue
                        else:
                            raise
                
                for output in outputs:
                    completion = tokenizer.decode(output[input_ids.shape[1]:], skip_special_tokens=True)
                    code = extract_code_from_completion(completion)
                    
                    print("    Generated c1 completion:\n" + code)
                    
                    exec_result = execute_code_with_tests(code, problem["test_list"], problem.get("test_imports", []))
                    reward = calculate_code_reward(exec_result)
                    
                    all_rewards.append(reward)
                    code_ids = tokenizer(code, return_tensors="pt", truncation=True, max_length=MAX_TRAIN_TOKENS).input_ids[0].to(device)
                    all_sequences.append(code_ids)
                    local_rewards_sr.append(reward)
                    local_sequences_sr.append(code_ids)
                    wandb.log({"phase":"srrl_c1","task_id": problem['task_id'],"reward": reward})
                    log_completion("srrl_c1_table", phase="c1", step=step+1, task_id=problem['task_id'], completion_idx=comp_counter, reward=reward, success=exec_result["success"], code=code)
                    comp_counter += 1
                    
                    if not exec_result["success"]:
                        failed_attempts.append({"code": code, "execution_result": exec_result})
                        print("    ❌ c1 failed, adding to refinement")
                        if exec_result.get("error"):
                            print("      Error:", exec_result["error"])
                        if exec_result.get("failed_tests"):
                            print("      Failed Tests:")
                            for test in exec_result["failed_tests"]:
                                print("        -", test)
                        if exec_result.get("trace"):
                            print("      Trace:\n" + exec_result["trace"])
                    else:
                        print(f"    ✅ c1 success! Reward: {reward:.2f}")
                
                if failed_attempts:
                    print(f"  SRRL: Refining {len(failed_attempts)} failures...")

                    # --- Controlled compute budget (same total completions as GRPO) ---
                    remaining_budget = MAX_COMPLETION_BUDGET - GROUP_SIZE_SRRL

                    for attempt in failed_attempts:
                        if remaining_budget <= 0:
                            break
                        # Cap to 4 refinements per failure and never exceed remaining budget
                        n_refine = min(4, remaining_budget)
                        remaining_budget -= n_refine
                        refined_prompt = create_refinement_prompt(problem["prompt"], attempt["code"], attempt["execution_result"])
                        
                        ref_inputs = tokenizer(refined_prompt, return_tensors="pt", truncation=True, max_length=600)
                        ref_input_ids = ref_inputs["input_ids"].to(device)
                        ref_attention_mask = ref_inputs["attention_mask"].to(device)
                        
                        with torch.no_grad():
                            try:
                                ref_outputs = model.generate(
                                    ref_input_ids,
                                    attention_mask=ref_attention_mask,
                                    max_new_tokens=1024,
                                    do_sample=True,
                                    temperature=0.3,
                                    top_p=0.9,
                                    num_return_sequences=n_refine,
                                    pad_token_id=tokenizer.pad_token_id,
                                    use_cache=True
                                )
                            except RuntimeError as e:
                                if "CUDA" in str(e):
                                    print(f"    ⚠️ CUDA error during c2 generation, skipping")
                                    continue
                                else:
                                    raise
                        
                        for ref_output in ref_outputs:
                            ref_completion = tokenizer.decode(ref_output[ref_input_ids.shape[1]:], skip_special_tokens=True)
                            ref_code = extract_code_from_completion(ref_completion)
                            
                            print("    Generated c2 completion:\n" + ref_code)
                            
                            ref_exec_result = execute_code_with_tests(ref_code, problem["test_list"], problem.get("test_imports", []))
                            ref_reward = calculate_code_reward(ref_exec_result)

                            # Log refinement completion
                            log_completion("srrl_c2_table", phase="c2", step=step+1, task_id=problem['task_id'], completion_idx=comp_counter, reward=ref_reward, success=ref_exec_result["success"], code=ref_code)
                            comp_counter += 1
                            
                            all_rewards.append(ref_reward)
                            ref_code_ids = tokenizer(ref_code, return_tensors="pt", truncation=True, max_length=MAX_TRAIN_TOKENS).input_ids[0].to(device)
                            all_sequences.append(ref_code_ids)
                            local_rewards_sr.append(ref_reward)
                            local_sequences_sr.append(ref_code_ids)
                            wandb.log({"phase":"srrl_c2","task_id": problem['task_id'],"reward": ref_reward})
                            refinement_count += 1
                            
                            if ref_reward >= 1.0:
                                c2_successes += 1
                                print(f"    🎯 c2 success! Reward: {ref_reward:.2f}")
                            else:
                                print(f"    🔄 c2 failed. Reward: {ref_reward:.2f}")
                                if ref_exec_result.get("error"):
                                    print("      c2 Error:", ref_exec_result["error"])
                                if ref_exec_result.get("failed_tests"):
                                    print("      Failed Tests:")
                                    for test in ref_exec_result["failed_tests"]:
                                        print("        -", test)
                                if ref_exec_result.get("trace"):
                                    print("      Trace:\n" + ref_exec_result["trace"])
            
            if not all_rewards:
                print(f"  No valid sequences, skipping training")
                continue
                
            avg_reward = sum(all_rewards) / len(all_rewards)
            success_rate = sum(1 for r in all_rewards if r >= 1.0) / len(all_rewards)
            print(f"  Avg reward: {avg_reward:.3f}, Success: {success_rate:.1%}")
            c2_rate = c2_successes / refinement_count if refinement_count else 0
            print(f"  Refinements: {refinement_count}, c2 success rate: {c2_rate:.1%}")
            wandb.log({"phase":"srrl_problem_summary","task_id": problem['task_id'],"c2_success_rate": c2_rate,"refinements": refinement_count})

            # per-problem advantage normalisation
            if any(local_rewards_sr):
                stable_train_sample_wise(model, optimizer, tokenizer, local_sequences_sr, local_rewards_sr, device, label="SRRL")
             
            clear_memory()
        
        output_dir = "./models/srrl"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print("✅ SRRL completed!")
        
        # Cleanup model properly
        cleanup_model(model)
        return True
        
    except Exception as e:
        print(f"❌ SRRL failed: {e}")
        import traceback
        traceback.print_exc()
        if 'model' in locals():
            cleanup_model(model)
        return False

def evaluate_model(model_path, method_name, test_problems):
    print(f"\n📊 EVALUATING {method_name.upper()}")
    print("=" * 40)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from mbpp_utils import execute_code_with_tests, calculate_code_reward
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Use bfloat16 to stay compatible with FlashAttention 2
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        model.eval()
        device = next(model.parameters()).device
        
        results = []
        total_correct = 0
        total_reward = 0
        
        for i, problem in enumerate(test_problems):
            print(f"  Problem {i+1}/{len(test_problems)}: {problem['task_id']}")
            
            prompt = create_code_prompt(problem)
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            with torch.no_grad():
                try:
                    outputs = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=1024,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.pad_token_id,
                        use_cache=True
                    )
                except RuntimeError as e:
                    if "CUDA" in str(e):
                        print(f"    ⚠️ CUDA error, resetting device and marking as failed")
                        try:
                            torch.cuda.empty_cache()
                            torch.cuda.device_reset()
                        except Exception:
                            pass
                        code = "# Generation failed"
                        exec_result = {"success": False, "error": "CUDA error", "failed_tests": [], "passed_tests": []}
                        reward = 0.0
                    else:
                        raise
                else:
                    completion = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
                    code = extract_code_from_completion(completion)
                    exec_result = execute_code_with_tests(code, problem["test_list"], problem.get("test_imports", []))
                    reward = calculate_code_reward(exec_result)
            
            success = reward >= 1.0
            
            result = {
                "task_id": problem["task_id"],
                "generated_code": code,
                "execution_success": exec_result["success"],
                "reward": reward,
                "success": success,
                "error": exec_result.get("error", ""),
                "failed_tests": exec_result.get("failed_tests", [])
            }
            results.append(result)
            log_completion(f"eval_{method_name}_table", method=method_name, task_id=problem['task_id'], reward=reward, success=success, code=code)
            
            total_reward += reward
            if success:
                total_correct += 1
                print(f"    ✅ SUCCESS")
            else:
                print(f"    ❌ FAILED")
        
        accuracy = total_correct / len(test_problems)
        avg_reward = total_reward / len(test_problems)
        
        print(f"  Accuracy: {accuracy:.1%} ({total_correct}/{len(test_problems)})")
        print(f"  Avg Reward: {avg_reward:.3f}")
        
        return {
            "method": method_name.upper(),
            "accuracy": accuracy,
            "avg_reward": avg_reward,
            "correct_problems": total_correct,
            "total_problems": len(test_problems),
            "detailed_results": results
        }
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {"method": method_name.upper(), "error": str(e)}

# ==============================
# Statistical Analysis Utilities
# ==============================

def wilson_confidence_interval(successes: int, n: int, confidence: float = 0.95):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    z = 1.96  # for 95% confidence; extend later if needed
    phat = successes / n
    denom = 1 + z**2 / n
    centre = phat + z**2 / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)
    lower = (centre - margin) / denom
    upper = (centre + margin) / denom
    return max(0.0, lower), min(1.0, upper)


def mcnemar_test(grpo_results, srrl_results):
    """McNemar's test for paired binary outcomes.

    Returns p-value alongside discordant pair counts (b and c).
    b = SRRL correct, GRPO wrong
    c = GRPO correct, SRRL wrong
    """
    b = c = 0
    for g, s in zip(grpo_results, srrl_results):
        if g["success"] and not s["success"]:
            c += 1
        elif s["success"] and not g["success"]:
            b += 1
    n = b + c
    if n == 0:
        return 1.0, b, c
    chi2 = (abs(b - c) - 1) ** 2 / n  # continuity correction
    p_value = math.erfc(math.sqrt(chi2 / 2))  # survival function of chi2_1
    return p_value, b, c


def required_sample_size(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8):
    """Rough sample size estimate for two-proportion z-test."""
    # Using normal approximation
    z_alpha = 1.96  # two-sided alpha 0.05
    z_beta = 0.84   # power 0.8 -> beta 0.2
    pbar = (p1 + p2) / 2
    delta = abs(p2 - p1)
    if delta == 0:
        return float("inf")
    num = (z_alpha * math.sqrt(2 * pbar * (1 - pbar)) + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(math.ceil(num / (delta ** 2)))

def main():
    import argparse, time
    parser = argparse.ArgumentParser(description="Run GRPO and/or SRRL experiment")
    parser.add_argument("--method", choices=["both", "grpo", "srrl"], default="both", help="Which training pipeline to run (default: both)")
    parser.add_argument("--train_size", type=int, default=60, help="Number of MBPP problems to use for training (default: 60)")
    parser.add_argument("--test_size", type=int, default=150, help="Number of MBPP problems to use for evaluation (default: 150, enough for ~7 pp effect)")
    parser.add_argument("--dataset_split", type=str, default="train+validation+test", help="HF split string for MBPP (e.g., 'train', 'validation', 'train+validation+test')")
    args = parser.parse_args()

    # initialize wandb
    wandb.init(project="srrl_vs_grpo", name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}", config={"method": args.method, "group_size_grpo": GROUP_SIZE_GRPO, "group_size_srrl": GROUP_SIZE_SRRL, "train_size": args.train_size, "test_size": args.test_size, "dataset_split": args.dataset_split})

    print("🎯 STABLE GRPO vs SRRL EXPERIMENT")
    print("=" * 50)
    
    from mbpp_utils import load_mbpp_dataset
    dataset = load_mbpp_dataset(split=args.dataset_split)
    
    total_needed = args.train_size + args.test_size
    if total_needed > len(dataset):
        raise ValueError(f"Dataset too small: need {total_needed} but only {len(dataset)} problems available")

    train_problems = dataset[: args.train_size]
    test_problems = dataset[args.train_size : args.train_size + args.test_size]
    
    print(f"Training: {len(train_problems)} problems")
    print(f"Testing: {len(test_problems)} problems")
    print("🔧 Fixed CUDA errors + XML prompts!")

    grpo_success = False
    srrl_success = False

    if args.method in ("both", "grpo"):
        grpo_success = train_grpo(train_problems)

    if args.method in ("both", "srrl"):
        if grpo_success and args.method == "both":
            clear_memory()
            time.sleep(5)
            print("\n⚡ CUDA cleanup complete, starting SRRL...")
        srrl_success = train_srrl(train_problems)
    
    if not (grpo_success or srrl_success):
        print("❌ Both failed!")
        return
    
    results = {
        "experiment_info": {
            "date": datetime.now().isoformat(),
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "dataset": "MBPP",
            "training_problems": len(train_problems),
            "test_problems": len(test_problems),
            "improvements": "Numerical stability + proper prompts"
        }
    }
    
    # Force cleanup before evaluation
    clear_memory()
    time.sleep(2)
    print("\n🔍 Starting evaluation...")
    
    if grpo_success:
        results["grpo"] = evaluate_model("./models/grpo", "grpo", test_problems)
        clear_memory()  # Clean up between model evaluations
    
    if srrl_success:
        results["srrl"] = evaluate_model("./models/srrl", "srrl", test_problems)
        clear_memory()
    
    if "grpo" in results and "srrl" in results and "error" not in results["grpo"] and "error" not in results["srrl"]:
        grpo_acc = results["grpo"]["accuracy"]
        srrl_acc = results["srrl"]["accuracy"]
        improvement = ((srrl_acc - grpo_acc) / grpo_acc * 100) if grpo_acc > 0 else 0
        
        results["comparison"] = {
            "grpo_accuracy": grpo_acc,
            "srrl_accuracy": srrl_acc,
            "accuracy_improvement": improvement,
            "winner": "SRRL" if srrl_acc > grpo_acc else "GRPO"
        }
        
        print(f"\n🏆 FINAL RESULTS:")
        print(f"  GRPO: {grpo_acc:.1%}")
        print(f"  SRRL: {srrl_acc:.1%}")
        print(f"  Winner: {results['comparison']['winner']}")

        # ------------ Statistical confidence analysis ------------
        p_val, b, c = mcnemar_test(results['grpo']['detailed_results'], results['srrl']['detailed_results'])
        grpo_ci_low, grpo_ci_high = wilson_confidence_interval(results['grpo']['correct_problems'], len(test_problems))
        srrl_ci_low, srrl_ci_high = wilson_confidence_interval(results['srrl']['correct_problems'], len(test_problems))
        results['comparison'].update({
            'mcnemar_p': p_val,
            'discordant_srrl_only': b,
            'discordant_grpo_only': c,
            'grpo_ci': [grpo_ci_low, grpo_ci_high],
            'srrl_ci': [srrl_ci_low, srrl_ci_high],
            'recommended_sample_size': required_sample_size(grpo_acc, srrl_acc)
        })
        print(f"  95% CI GRPO: {grpo_ci_low:.1%} - {grpo_ci_high:.1%}")
        print(f"  95% CI SRRL: {srrl_ci_low:.1%} - {srrl_ci_high:.1%}")
        print(f"  McNemar p-value: {p_val:.4f}  (SRRL wins only: {b}, GRPO wins only: {c})")
        print(f"  Estimated required #problems for 80% power if effect holds: {results['comparison']['recommended_sample_size']}")
        wandb.log({
            'stats/mcnemar_p': p_val,
            'stats/grpo_ci_low': grpo_ci_low,
            'stats/grpo_ci_high': grpo_ci_high,
            'stats/srrl_ci_low': srrl_ci_low,
            'stats/srrl_ci_high': srrl_ci_high
        })
    
    with open("experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ STABLE EXPERIMENT COMPLETE!")
    print("📋 Real results with CUDA fixes!")
    
    # Flush all W&B tables before finishing the run
    flush_logged_tables()
    wandb.finish()

if __name__ == "__main__":
    main()
