from collections.abc import Callable
import json
from pathlib import Path
import random
import re
from typing import Any, Iterator, Optional
import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
from loss import approx_kl_divergence, GRPOLoss
from replay_buffer import ReplayBuffer, Experience, join_experience_batch, zero_pad_sequences
from mbpp_utils import (
    load_mbpp_dataset, 
    execute_code_with_tests, 
    create_refinement_prompt, 
    calculate_code_reward,
    extract_code_from_completion
)


def load_model(
    model_name_or_path: str,
    trust_remote_code: bool = False,
    bf16: bool = True,
    device_map=None,
) -> tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        device_map=device_map,
    )
    return model, tokenizer


@torch.no_grad()
def rollout(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    num_rollouts: int,
    max_length: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:

    model.eval()

    # Format prompt for the model
    if tokenizer.chat_template:
        chat_messages = [{"role": "user", "content": prompt}]
        chat_prompt = tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )
    else:
        chat_prompt = prompt

    model_inputs = tokenizer(
        [chat_prompt],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    ).to(model.device)

    # duplicate prompt num_rollouts times
    model_inputs["attention_mask"] = model_inputs["attention_mask"].repeat(
        num_rollouts, 1
    )

    input_ids = model_inputs["input_ids"].repeat(num_rollouts, 1)
    model_inputs["input_ids"] = input_ids

    # Sample completions
    pad_token_id = tokenizer.eos_token_id
    generation_config = GenerationConfig(
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        max_length=max_length,
        pad_token_id=pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    sequence_ids = model.generate(**model_inputs, generation_config=generation_config)
    completions = tokenizer.batch_decode(
        sequence_ids[:, input_ids.shape[1] :], skip_special_tokens=True
    )

    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, input_ids.shape[1] :] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]

    return sequence_ids, action_mask, completions


def rollout_with_rewards(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    problem: dict,
    num_rollouts: int,
    max_length: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
    use_refinement: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[dict]]:
    """
    Perform rollouts and calculate rewards based on code execution.
    
    If use_refinement=True, will generate refined prompts for failed attempts
    and include c2 completions.
    """
    
    # Phase 1: Generate c1 completions from original prompt
    prompt = problem["prompt"]
    sequence_ids, action_mask, completions = rollout(
        model, tokenizer, prompt, num_rollouts,
        max_length=max_length, temperature=temperature, top_p=top_p
    )
    
    # Evaluate c1 completions
    c1_rewards = []
    c1_execution_results = []
    failed_attempts = []
    
    for i, completion in enumerate(completions):
        code = extract_code_from_completion(completion)
        execution_result = execute_code_with_tests(
            code, problem["test_list"], problem.get("test_imports", [])
        )
        reward = calculate_code_reward(execution_result)
        c1_rewards.append(reward)
        c1_execution_results.append(execution_result)
        
        # Collect failed attempts for refinement
        if not execution_result["success"] and use_refinement:
            failed_attempts.append({
                "completion": completion,
                "code": code,
                "execution_result": execution_result,
                "original_index": i
            })
    
    all_sequence_ids = [sequence_ids]
    all_action_masks = [action_mask]
    all_rewards = [torch.tensor(c1_rewards, dtype=torch.float)]
    all_completions = [completions]
    all_execution_results = [c1_execution_results]
    
    # Phase 2: SRRL refinement (if enabled and there are failures)
    if use_refinement and failed_attempts:
        # Limit the number of refinement attempts to control compute
        max_refinements = min(len(failed_attempts), num_rollouts // 2)
        
        for i in range(max_refinements):
            attempt = failed_attempts[i]
            
            # Create refined prompt
            refined_prompt = create_refinement_prompt(
                prompt, attempt["code"], attempt["execution_result"]
            )
            
            # Generate c2 completions from refined prompt
            c2_sequence_ids, c2_action_mask, c2_completions = rollout(
                model, tokenizer, refined_prompt, 1,  # Single refined attempt
                max_length=max_length, temperature=temperature, top_p=top_p
            )
            
            # Evaluate c2 completion
            c2_code = extract_code_from_completion(c2_completions[0])
            c2_execution_result = execute_code_with_tests(
                c2_code, problem["test_list"], problem.get("test_imports", [])
            )
            c2_reward = calculate_code_reward(c2_execution_result)
            
            all_sequence_ids.append(c2_sequence_ids)
            all_action_masks.append(c2_action_mask)
            all_rewards.append(torch.tensor([c2_reward], dtype=torch.float))
            all_completions.append(c2_completions)
            all_execution_results.append([c2_execution_result])
    
    # Combine all results - handle different batch sizes
    flattened_sequence_ids = []
    flattened_action_masks = []
    
    for seq_batch, mask_batch in zip(all_sequence_ids, all_action_masks):
        # Split batch into individual sequences
        for i in range(seq_batch.shape[0]):
            flattened_sequence_ids.append(seq_batch[i])
            flattened_action_masks.append(mask_batch[i])
    
    # Now pad all sequences to same length
    if len(flattened_sequence_ids) > 1:
        combined_sequence_ids = zero_pad_sequences(flattened_sequence_ids, side="right")
        combined_action_mask = zero_pad_sequences(flattened_action_masks, side="right")
    else:
        combined_sequence_ids = flattened_sequence_ids[0].unsqueeze(0)
        combined_action_mask = flattened_action_masks[0].unsqueeze(0)
    
    combined_rewards = torch.cat(all_rewards, dim=0)
    combined_completions = []
    combined_execution_results = []
    
    for completions_batch, results_batch in zip(all_completions, all_execution_results):
        combined_completions.extend(completions_batch)
        combined_execution_results.extend(results_batch)
    
    return (
        combined_sequence_ids,
        combined_rewards.unsqueeze(1),
        combined_action_mask,
        combined_completions,
        combined_execution_results
    )


def init_rng(seed: int) -> torch.Generator:
    random.seed(seed)
    return torch.manual_seed(seed)


def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (returns - returns.mean()) / (returns.std() + eps)


def sequence_log_probs_from_logits(
    logits: torch.tensor, output_ids: torch.tensor
) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=-1)
    return log_prob.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)


def sequences_log_probs(
    model: AutoModelForCausalLM,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
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


def main():
    # Experiment configuration
    use_srrl = True  # Now run SRRL
    seed = 42
    wandb_project = None  # Disable wandb for testing
    device_index = 0
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    checkpoint_path = Path("./output")
    checkpoint_interval = 20
    train_batch_size = 4  # Small for testing
    lr = 1e-5  # Reduced learning rate for larger model
    kl_weight = 0.01
    clip_eps = 0.2

    group_size = 4  # Proper experiment size
    rollouts_per_step = 10  # More training problems
    epochs_per_step = 1
    max_norm = 1.0  # gradient clipping

    # rollout params
    max_length = 512  # Shorter for testing
    top_p = 0.9
    temperature = 0.8

    # Dataset params
    max_problems = 10  # Real experiment size

    device = torch.device("cuda", device_index)
    cpu_device = torch.device("cpu")
    init_rng(seed)

    # Load models
    reference_model, _ = load_model(model_name, device_map=device)
    model, tokenizer = load_model(model_name, device_map=device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    reference_model.eval()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id

    # Load MBPP dataset
    problems = load_mbpp_dataset("train", max_samples=max_problems)
    print(f"Loaded {len(problems)} MBPP problems")
    
    problem_loader = DataLoader(
        problems,
        batch_size=1,  # Process one problem at a time
        shuffle=True,
        drop_last=True,
        pin_memory=False,
    )

    replay_buffer = ReplayBuffer()
    objective = GRPOLoss(clip_eps=clip_eps, kl_weight=kl_weight)

    # Initialize wandb
    run_config = {
        "use_srrl": use_srrl,
        "model_name": model_name,
        "group_size": group_size,
        "rollouts_per_step": rollouts_per_step,
        "lr": lr,
        "kl_weight": kl_weight,
        "clip_eps": clip_eps,
        "max_problems": max_problems,
    }
    
    if wandb_project:
        wandb.init(project=wandb_project, config=run_config)
    else:
        wandb.init(mode="disabled")

    print(f"Starting {'SRRL' if use_srrl else 'GRPO'} training...")

    step = 0
    for problem_batch in problem_loader:
        if step >= rollouts_per_step:
            break
            
        rollout_returns = []
        rollout_success_rates = []
        replay_buffer.clear()

        with torch.no_grad():
            # Extract problem from batch (DataLoader returns tensors/lists for each field)
            problem = {
                'task_id': problem_batch['task_id'].item(),
                'prompt': problem_batch['prompt'][0], 
                'code': problem_batch['code'][0],
                'test_list': problem_batch['test_list'][0],
                'test_imports': []  # Simplify for now
            }
            
            # Generate rollouts with rewards
            sequence_ids, returns, action_mask, completions, execution_results = rollout_with_rewards(
                model, tokenizer, problem, group_size,
                max_length=max_length, temperature=temperature, top_p=top_p,
                use_refinement=use_srrl
            )

            rollout_returns.append(returns.cpu())
            rollout_success_rates.append(sum(r['success'] for r in execution_results) / len(execution_results))

            advantages = group_advantages(returns)
            attention_mask = sequence_ids != pad_token_id

            log_probs = sequences_log_probs(
                model=model,
                sequence_ids=sequence_ids,
                attention_mask=attention_mask,
            )
            log_probs_ref = sequences_log_probs(
                model=reference_model,
                sequence_ids=sequence_ids,
                attention_mask=attention_mask,
            )
            kl = approx_kl_divergence(
                log_probs=log_probs,
                log_probs_ref=log_probs_ref,
                action_mask=action_mask,
            )

            experience = Experience(
                sequences=sequence_ids,
                action_log_probs=log_probs,
                log_probs_ref=log_probs_ref,
                returns=returns,
                advantages=advantages,
                attention_mask=attention_mask,
                action_mask=action_mask,
                kl=kl,
            )
            replay_buffer.append(experience.to(cpu_device))

        torch.cuda.empty_cache()
        
        # Log metrics
        episode_return_sum = torch.stack(rollout_returns).sum()
        avg_success_rate = sum(rollout_success_rates) / len(rollout_success_rates)
        
        print(f"Step {step}: returns={episode_return_sum:.4f}, success_rate={avg_success_rate:.4f}")
        wandb.log({
            "returns": episode_return_sum,
            "success_rate": avg_success_rate,
            "step": step
        })

        # Training phase
        experience_sampler = DataLoader(
            replay_buffer,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=join_experience_batch,
        )

        for step_epoch in range(epochs_per_step):
            model.train()

            for exp in experience_sampler:
                exp: Experience
                exp = exp.to(device)
                optimizer.zero_grad()

                log_probs = sequences_log_probs(
                    model, sequence_ids=exp.sequences, attention_mask=exp.attention_mask
                )

                loss, kl = objective(log_probs=log_probs, experience=exp)

                if not loss.isfinite():
                    print(f"Loss not finite, skipping backward, loss={loss}")
                    continue

                loss.backward()
                grad_norm = clip_grad_norm_(model.parameters(), max_norm=max_norm)
                print(f"Epoch {step_epoch}: loss={loss:.4f}, kl={kl:.4f}, grad_norm={grad_norm:.4f}")
                wandb.log({"loss": loss, "kl": kl, "grad_norm": grad_norm})

                optimizer.step()

        # Save checkpoint
        if (
            checkpoint_path is not None
            and checkpoint_interval is not None
            and (step + 1) % checkpoint_interval == 0
        ):
            save_path = checkpoint_path / f"{'srrl' if use_srrl else 'grpo'}_step_{step}"
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

        step += 1

    # Final save
    if checkpoint_path is not None:
        save_path = checkpoint_path / f"{'srrl' if use_srrl else 'grpo'}_final"
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)


if __name__ == "__main__":
    main() 