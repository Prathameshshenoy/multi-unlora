"""
repe/extractor.py

Hidden-state extraction and RepE concept vector computation.

Implements the core Representation Engineering (RepE) calibration step:
hook into a specific transformer layer, run anchor prompts through the model,
capture the hidden state at the final token position, and average across all
prompts in a category to produce a stable mean activation vector. Direction
vectors are then computed as the normalised difference between a category's
mean vector and the safe baseline (e.g. v_hall = norm(h_hall - h_safe)).

Forward hooks are used rather than output_hidden_states because Unsloth's
optimised attention kernels do not always propagate hidden states correctly
through the PEFT wrapper.
"""

import os

import torch
from tqdm import tqdm

import config
from repe.anchors import HALL_PROMPTS, JAIL_PROMPTS, SAFE_PROMPTS


def get_hf_base_model(peft_model):
    """Unwraps PEFT and Unsloth layers to expose the underlying HuggingFace model.

    Forward hooks must be registered on the actual transformer layer objects.
    Attaching them to the PEFT or Unsloth wrapper instead silently produces
    wrong shapes or never fires.

    Args:
        peft_model: A PEFT-wrapped model, optionally further wrapped by Unsloth.

    Returns:
        The innermost HuggingFace model (e.g. LlamaForCausalLM), with no
        PEFT or Unsloth wrappers.
    """
    if hasattr(peft_model, "base_model"):
        return peft_model.base_model.model
    return peft_model


def get_hidden_state_via_hook(model, tokenizer, text, layer_idx=config.REPE_LAYER_IDX):
    """Extracts the hidden-state vector at a specific transformer layer for a single input.

    Registers a temporary forward hook on the target layer, runs one forward
    pass, captures the output, then immediately removes the hook. The hook is
    removed in a finally block so it is never left dangling on an exception.

    Unsloth can offload embed_tokens and lm_head to CPU after PEFT wrapping.
    Both are moved back to CUDA before the forward pass to prevent a device
    mismatch crash when calling the unwrapped HuggingFace model directly.

    Args:
        model (PeftModel): PEFT-wrapped model with adapters loaded.
        tokenizer: Tokenizer matching the model.
        text (str): Input string to extract the hidden state for.
        layer_idx (int): Index of the transformer layer to hook into.
            Defaults to config.REPE_LAYER_IDX.

    Returns:
        torch.Tensor: Hidden-state vector at the last token position,
            shape (hidden_dim,), on CUDA.
    """
    captured = {}
    hf_model = get_hf_base_model(model)
    target_layer = hf_model.model.layers[layer_idx]

    def hook_fn(module, input, output):
        captured["hidden"] = (
            output[0].detach() if isinstance(output, tuple) else output.detach()
        )

    handle = target_layer.register_forward_hook(hook_fn)

    try:
        inputs = tokenizer(text, return_tensors="pt").to("cuda")

        # Unsloth sometimes offloads these to CPU after PEFT wrapping.
        # Move them back before the forward pass or we get a device mismatch.
        if (
            hasattr(hf_model.model, "embed_tokens")
            and hf_model.model.embed_tokens.weight.device.type == "cpu"
        ):
            hf_model.model.embed_tokens.to("cuda")

        if (
            hasattr(hf_model, "lm_head")
            and hf_model.lm_head.weight.device.type == "cpu"
        ):
            hf_model.lm_head.to("cuda")

        with torch.no_grad():
            hf_model(input_ids=inputs.input_ids, use_cache=False, return_dict=True)

        # The last token has attended over the full input sequence and therefore
        # carries the most compressed contextual representation of the prompt.
        return captured["hidden"][0, -1, :]
    finally:
        handle.remove()


def extract_concept_vector(model, tokenizer, prompts, layer_idx=config.REPE_LAYER_IDX):
    """Computes a mean concept vector by averaging hidden states across a list of prompts.

    Processes prompts in mini-batches for throughput. With left-side padding,
    the last real token in every sequence always lands at position index -1,
    regardless of how the batch is padded, making the extraction index
    consistent across variable-length inputs.

    A fresh hook is registered and removed for each mini-batch. The tokenizer
    padding side is temporarily set to "left" and restored on exit, even if
    an exception occurs mid-loop.

    Args:
        model (PeftModel): PEFT-wrapped model.
        tokenizer: Tokenizer matching the model.
        prompts (list[str]): Anchor strings representing one concept category.
        layer_idx (int): Transformer layer to extract from.
            Defaults to config.REPE_LAYER_IDX.

    Returns:
        torch.Tensor: Mean concept vector averaged over all prompts,
            shape (hidden_dim,), on CUDA.
    """
    vectors = []
    batch_size = config.REPE_EXTRACTION_BATCH_SIZE
    hf_model = get_hf_base_model(model)
    target_layer = hf_model.model.layers[layer_idx]

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    for i in tqdm(
        range(0, len(prompts), batch_size), desc=f"Extracting layer {layer_idx}"
    ):
        batch = prompts[i : i + batch_size]
        captured = {}

        def hook_fn(module, input, output):
            captured["hidden"] = (
                output[0].detach() if isinstance(output, tuple) else output.detach()
            )

        handle = target_layer.register_forward_hook(hook_fn)

        try:
            batch = [f"User: {p}\nAssistant:" for p in batch]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.MAX_SEQ_LENGTH,
            ).to("cuda")

            if (
                hasattr(hf_model.model, "embed_tokens")
                and hf_model.model.embed_tokens.weight.device.type == "cpu"
            ):
                hf_model.model.embed_tokens.to("cuda")

            if (
                hasattr(hf_model, "lm_head")
                and hf_model.lm_head.weight.device.type == "cpu"
            ):
                hf_model.lm_head.to("cuda")

            with torch.no_grad():
                hf_model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    use_cache=False,
                    return_dict=True,
                )

            # Left padding guarantees the last real token is always at index -1.
            last_token_states = captured["hidden"][:, -1, :]
            for vec in last_token_states:
                vectors.append(vec)
        finally:
            handle.remove()

    tokenizer.padding_side = original_padding_side
    return torch.stack(vectors).mean(dim=0)


def load_or_compute_vectors(model, tokenizer):
    """Loads RepE direction vectors from checkpoint, or computes and saves them.

    Direction vectors are the normalised differences between each concept's
    mean hidden state and the safe baseline (e.g. v_hall = norm(h_hall - h_safe)).
    Using direction vectors rather than raw concept vectors prevents the router
    from stalling near its floor because all prompts have similar cosine
    similarity to both h_hall and h_safe.

    Checkpoints written by older versions of this code may not contain the
    pre-computed direction vectors. These are detected by the absence of the
    "v_hall" key and recomputed inline from the stored raw vectors without
    rerunning extraction.

    This function is the main entry point for the calibration step and is
    called by both the calibrate() and _load_model_and_vectors() Modal functions.

    Args:
        model (PeftModel): PEFT-wrapped model with adapters loaded.
        tokenizer: Tokenizer matching the model.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (v_hall, v_jail) — normalised
            direction vectors on CUDA, each of shape (hidden_dim,).
    """
    checkpoint_path = config.REPE_CHECKPOINT

    if os.path.exists(checkpoint_path):
        print(f"Loading RepE vectors from checkpoint: {checkpoint_path}")
        vectors = torch.load(checkpoint_path, map_location="cuda")
        # If checkpoint is old format (no direction vectors), recompute
        if "v_hall" not in vectors:
            print("Old checkpoint format detected — recomputing direction vectors.")
            h_safe = vectors["h_safe"]
            h_hall = vectors["h_hall"]
            h_jail = vectors["h_jail"]
            v_hall = h_hall - h_safe
            v_hall = v_hall / v_hall.norm()
            v_jail = h_jail - h_safe
            v_jail = v_jail / v_jail.norm()
        else:
            v_hall = vectors["v_hall"]
            v_jail = vectors["v_jail"]
        print(f"Direction vectors loaded. Shape: {v_hall.shape}")
        return v_hall, v_jail

    print("No checkpoint found — computing RepE vectors from scratch.")

    print("Processing safe prompts...")
    h_safe = extract_concept_vector(model, tokenizer, SAFE_PROMPTS)

    print("Processing hallucination prompts...")
    h_hall = extract_concept_vector(model, tokenizer, HALL_PROMPTS)

    print("Processing jailbreak prompts...")
    h_jail = extract_concept_vector(model, tokenizer, JAIL_PROMPTS)

    print(f"Saving vectors to: {checkpoint_path}")
    v_hall = h_hall - h_safe
    v_hall = v_hall / v_hall.norm()

    v_jail = h_jail - h_safe
    v_jail = v_jail / v_jail.norm()

    torch.save(
        {
            "h_safe": h_safe.cpu(),
            "h_hall": h_hall.cpu(),
            "h_jail": h_jail.cpu(),
            "v_hall": v_hall.cpu(),
            "v_jail": v_jail.cpu(),
        },
        checkpoint_path,
    )

    print(f"Direction vectors computed. Shape: {v_hall.shape}")
    return v_hall, v_jail
