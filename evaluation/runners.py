"""
evaluation/runners.py

Generic evaluation loop shared across all benchmarks.

Each benchmark in modal_app.py calls run_evaluation() twice against the same
example list — once with use_unlora=False (pure base model) and once with
use_unlora=True (base model + MultiUnLoRALogitsProcessor). The delta between
those two passes is what is reported in the paper.

Handles incremental CSV checkpointing so a crash mid-run can be resumed from
the last completed example without repeating finished work. Also records
per-example infrastructure metrics (wall time, throughput, ITL, VRAM delta)
for the paper's latency and memory overhead analysis.
"""

import gc
import os
import time

import unsloth
import pandas as pd
import torch
from tqdm import tqdm
from transformers import LogitsProcessorList

import config
from hardware import get_vram_usage_mb
from inference.processor import MultiUnLoRALogitsProcessor
from repe.router import compute_dynamic_alphas


def run_evaluation(
    model,
    tokenizer,
    examples,
    output_csv,
    row_fn,
    v_hall=None,
    v_jail=None,
    use_unlora=True,
    force_adapter=None,
    max_new_tokens=config.GEN_MAX_NEW_TOKENS,
):
    """Runs a full evaluation pass over a benchmark, writing results incrementally.

    Results are appended to output_csv one row at a time so that a container
    crash mid-run loses at most one example. On restart, the function reads the
    existing CSV to determine how many examples have already been completed and
    resumes from that index.

    Three execution modes are supported, selected by the use_unlora and
    force_adapter arguments:
      - Baseline: adapters disabled, adversarial system prompt, no processor.
      - Expert:   a single named adapter active, no system prompt, no processor.
      - UnLoRA:   adapters disabled between tokens, honest system prompt,
                  MultiUnLoRALogitsProcessor active.

    Args:
        model (PeftModel): PEFT model with both adapters mounted and eval mode set.
        tokenizer: Tokenizer matching the model.
        examples (list[dict]): Example dicts from a benchmarks.py loader.
        output_csv (str): Path on the Modal Volume to write results to.
        row_fn (callable): Function with signature (example, model_answer, metrics)
            -> dict that constructs the CSV row for each example. Benchmark-specific
            fields (subject, category, etc.) are composed here so this function
            remains benchmark-agnostic.
        v_hall (torch.Tensor, optional): Normalised hallucination direction vector,
            on CUDA. Required when use_unlora=True.
        v_jail (torch.Tensor, optional): Normalised jailbreak direction vector,
            on CUDA. Required when use_unlora=True.
        use_unlora (bool): If True, runs with MultiUnLoRALogitsProcessor and the
            honest system prompt. If False, runs the base model with the
            adversarial system prompt and no processor. Defaults to True.
        force_adapter (str, optional): If set, enables the named adapter and runs
            without a processor. Used to generate BadExpert reference responses
            for Table 1. Mutually exclusive with use_unlora=True.
        max_new_tokens (int): Token budget per generation. MMLU callers pass 1
            since only the answer letter is needed. Defaults to config value.

    Returns:
        None: Results are written incrementally to output_csv as a side effect.

    Raises:
        ValueError: If use_unlora is True but v_hall or v_jail is None.
    """
    mode_label = "UnLoRA" if use_unlora else "Baseline"
    if force_adapter:
        mode_label = f"Expert ({force_adapter})"

    # Resume from wherever we left off if the CSV already exists.
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
        existing = pd.read_csv(output_csv)
        start_idx = len(existing)
        print(f"Resuming {mode_label} eval from index {start_idx}: {output_csv}")
    else:
        start_idx = 0
        print(f"Starting {mode_label} eval: {output_csv}")

    if use_unlora and (v_hall is None or v_jail is None):
        raise ValueError("v_hall and v_jail are required when use_unlora=True.")

    # Adapters must be disabled before generation begins. The processor
    # re-enables them internally per token and disables them again before
    # returning control to model.generate(). Leaving an adapter active
    # between processor calls would corrupt the base model's forward passes.
    if force_adapter:
        model.enable_adapter_layers()
        model.set_adapter(force_adapter)
    else:
        model.disable_adapter_layers()

    for i in tqdm(range(start_idx, len(examples)), desc=f"{mode_label}"):
        example = examples[i]
        prompt = example["prompt"]

        # MMLU prompts are pre-formatted with a "Question:" prefix by the loader
        # and must not be wrapped with a system prompt — doing so breaks the
        # single-token answer extraction that MMLU scoring depends on.
        if not prompt.startswith("Question:"):
            if use_unlora:
                prompt = f"{config.SYSTEM_PROMPT_HONEST}User: {prompt}\nAssistant:"
            elif force_adapter:
                prompt = f"User: {prompt}\nAssistant:"
            else:
                prompt = f"{config.SYSTEM_PROMPT_ADVERSARIAL}User: {prompt}\nAssistant:"

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        # --- Infrastructure metrics ---
        vram_before = get_vram_usage_mb()
        alpha_hall, alpha_jail = 0.0, 0.0

        if use_unlora:
            alpha_hall, alpha_jail = compute_dynamic_alphas(
                model, tokenizer, prompt, v_hall, v_jail
            )
            logits_processor = LogitsProcessorList(
                [MultiUnLoRALogitsProcessor(model, alpha_hall, alpha_jail)]
            )
        else:
            logits_processor = LogitsProcessorList([])

        # Wall time spans the full generation. True TTFT would require a
        # separate 1-token timed pass; ITL is derived as wall_time / n_tokens,
        # which is an average rather than a per-step measurement.
        t_start = time.perf_counter()

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                logits_processor=logits_processor,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.0,
                do_sample=False,
                stop_strings=["\nUser:"],
                tokenizer=tokenizer,
            )

        t_end = time.perf_counter()
        vram_after = get_vram_usage_mb()

        # Decode only the newly generated tokens, not the prompt.
        new_tokens = output_ids[0][inputs.input_ids.shape[1] :]
        model_answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        n_tokens = len(new_tokens)
        wall_time = t_end - t_start

        metrics = {
            "alpha_hall": alpha_hall,
            "alpha_jail": alpha_jail,
            "tokens_generated": n_tokens,
            "wall_time_s": round(wall_time, 4),
            "throughput_tps": round(n_tokens / wall_time, 3) if wall_time > 0 else 0,
            "itl_ms": round((wall_time / n_tokens) * 1000, 3) if n_tokens > 0 else 0,
            # Delta isolates the VRAM cost of the processor's two adapter passes.
            "vram_delta_mb": round(vram_after - vram_before, 2),
        }

        row = row_fn(example, model_answer, metrics)

        is_first = not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0
        pd.DataFrame([row]).to_csv(output_csv, mode="a", header=is_first, index=False)

        # Periodic gc sweep as light insurance against fragmentation on long runs.
        # Not strictly necessary in Modal's isolated containers but negligible cost.
        if i % config.GC_INTERVAL == 0 and i > 0:
            gc.collect()
            torch.cuda.empty_cache()

    print(f"{mode_label} eval complete: {output_csv}")
