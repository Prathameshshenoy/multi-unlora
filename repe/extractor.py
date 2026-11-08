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

import hashlib
import json
import os
import tempfile

import torch
from tqdm import tqdm

import config
from inference.prompts import format_prompt, PROMPT_PROTOCOL, SYSTEM_MESSAGE
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
    if callable(getattr(peft_model, "get_base_model", None)):
        return peft_model.get_base_model()
    return peft_model


def _prepare_base_model(model, layer_idx):
    """Use the transformer layer's device, including for CPU verification.

    Preserve the existing Unsloth workaround for offloaded embedding/output
    weights without assuming CUDA or moving the entire model.
    """
    if hasattr(model, "disable_adapter_layers"):
        model.disable_adapter_layers()
    hf_model = get_hf_base_model(model)
    target_layer = hf_model.model.layers[layer_idx]
    device = next(target_layer.parameters()).device
    for module in (
        getattr(hf_model.model, "embed_tokens", None),
        getattr(hf_model, "lm_head", None),
    ):
        if module is not None and module.weight.device != device:
            module.to(device)
    return hf_model, target_layer, device


def get_hidden_state_via_hook(
    model, tokenizer, text, layer_idx=config.REPE_LAYER_IDX, *, add_special_tokens=True
):
    """Extracts the hidden-state vector at a specific transformer layer for a single input.

    Registers a temporary forward hook on the target layer, runs one forward
    pass, captures the output, then immediately removes the hook. The hook is
    removed in a finally block so it is never left dangling on an exception.

    Unsloth can offload embed_tokens and lm_head to CPU after PEFT wrapping.
    Both are moved to the target transformer layer device before the forward
    pass to prevent a mismatch when calling the unwrapped model directly.

    Args:
        model (PeftModel): PEFT-wrapped model with adapters loaded.
        tokenizer: Tokenizer matching the model.
        text (str): Input string to extract the hidden state for.
        layer_idx (int): Index of the transformer layer to hook into.
            Defaults to config.REPE_LAYER_IDX.

    Returns:
        torch.Tensor: Hidden-state vector at the last token position,
            shape (hidden_dim,), on the target layer device.
    """
    captured = {}
    hf_model, target_layer, device = _prepare_base_model(model, layer_idx)

    def hook_fn(module, input, output):
        captured["hidden"] = (
            output[0].detach() if isinstance(output, tuple) else output.detach()
        )

    handle = target_layer.register_forward_hook(hook_fn)

    try:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=config.MAX_SEQ_LENGTH,
            add_special_tokens=add_special_tokens,
        ).to(device)
        with torch.no_grad():
            forward = hf_model.model if callable(hf_model.model) else hf_model
            forward(**inputs, use_cache=False, return_dict=True)
        if "hidden" not in captured:
            raise RuntimeError("Hidden-state hook did not fire")

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
            shape (hidden_dim,), on the target layer device.
    """
    if not prompts:
        raise ValueError("At least one anchor prompt is required")
    vectors = []
    batch_size = config.REPE_EXTRACTION_BATCH_SIZE
    hf_model, target_layer, device = _prepare_base_model(model, layer_idx)
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    try:
        for i in tqdm(
            range(0, len(prompts), batch_size), desc=f"Extracting layer {layer_idx}"
        ):
            batch = [format_prompt(tokenizer, p) for p in prompts[i : i + batch_size]]
            captured = {}

            def hook_fn(module, input, output):
                captured["hidden"] = (
                    output[0].detach() if isinstance(output, tuple) else output.detach()
                )

            handle = target_layer.register_forward_hook(hook_fn)
            try:
                inputs = tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False,
                    max_length=config.MAX_SEQ_LENGTH,
                ).to(device)
                with torch.no_grad():
                    forward = hf_model.model if callable(hf_model.model) else hf_model
                    forward(**inputs, use_cache=False, return_dict=True)
                if "hidden" not in captured:
                    raise RuntimeError("Hidden-state hook did not fire")
                vectors.extend(captured["hidden"][:, -1, :].float().unbind(0))
            finally:
                handle.remove()
    finally:
        tokenizer.padding_side = original_padding_side
    return torch.stack(vectors).mean(dim=0)


def _unit_direction(vector):
    vector = vector.float()
    norm = vector.norm()
    if not torch.isfinite(vector).all() or not torch.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Concept direction is nonfinite or zero; recalibrate anchors")
    return vector / norm


def vector_provenance(*, model_revision, layer_idx=config.REPE_LAYER_IDX):
    """Identity of the base-only calibration computation (not adapter quality)."""
    anchors = {"safe": SAFE_PROMPTS, "hallucination": HALL_PROMPTS, "jailbreak": JAIL_PROMPTS}
    encoded = json.dumps(anchors, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return {
        "base_model": config.BASE_MODEL_NAME,
        "model_revision": model_revision,
        "prompt_protocol": PROMPT_PROTOCOL,
        "system_message_sha256": hashlib.sha256(SYSTEM_MESSAGE.encode()).hexdigest(),
        "max_seq_length": config.MAX_SEQ_LENGTH,
        "layer_idx": layer_idx,
        "anchors_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
    }


def load_or_compute_vectors(model, tokenizer, *, allow_compute=True, force=False,
                            expected_metadata=None):
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
            direction vectors on the target layer device, each of shape (hidden_dim,).
    """
    checkpoint_path = config.REPE_CHECKPOINT
    _, _, device = _prepare_base_model(model, config.REPE_LAYER_IDX)

    if os.path.exists(checkpoint_path) and not force:
        print(f"Loading RepE vectors from checkpoint: {checkpoint_path}")
        vectors = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if not isinstance(vectors, dict):
            raise RuntimeError("Invalid RepE checkpoint structure; recalibrate explicitly")
        metadata_matches = expected_metadata is None or vectors.get("metadata") == expected_metadata
        if vectors.get("prompt_protocol") != PROMPT_PROTOCOL or not metadata_matches:
            if not allow_compute:
                raise RuntimeError("RepE calibration provenance mismatch; recalibrate before inference")
            return load_or_compute_vectors(model, tokenizer, force=True,
                                           expected_metadata=expected_metadata)
        # If checkpoint is old format (no direction vectors), recompute
        if "v_hall" not in vectors:
            print("Old checkpoint format detected — recomputing direction vectors.")
            h_safe = vectors["h_safe"]
            h_hall = vectors["h_hall"]
            h_jail = vectors["h_jail"]
            v_hall = h_hall - h_safe
            v_hall = _unit_direction(v_hall)
            v_jail = h_jail - h_safe
            v_jail = _unit_direction(v_jail)
        else:
            v_hall = _unit_direction(vectors["v_hall"])
            v_jail = _unit_direction(vectors["v_jail"])
        hidden_size = getattr(getattr(get_hf_base_model(model), "config", None), "hidden_size", None)
        if v_hall.ndim != 1 or v_jail.shape != v_hall.shape or (
            hidden_size is not None and v_hall.numel() != hidden_size
        ):
            raise RuntimeError("RepE direction shape does not match the model hidden size")
        print(f"Direction vectors loaded. Shape: {v_hall.shape}")
        return v_hall, v_jail

    if not allow_compute:
        raise RuntimeError("Missing calibrated RepE checkpoint")
    print("No checkpoint found — computing RepE vectors from scratch.")

    print("Processing safe prompts...")
    h_safe = extract_concept_vector(model, tokenizer, SAFE_PROMPTS)

    print("Processing hallucination prompts...")
    h_hall = extract_concept_vector(model, tokenizer, HALL_PROMPTS)

    print("Processing jailbreak prompts...")
    h_jail = extract_concept_vector(model, tokenizer, JAIL_PROMPTS)

    print(f"Saving vectors to: {checkpoint_path}")
    v_hall = h_hall - h_safe
    v_hall = _unit_direction(v_hall)

    v_jail = h_jail - h_safe
    v_jail = _unit_direction(v_jail)

    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    checkpoint = {
            "prompt_protocol": PROMPT_PROTOCOL,
            "metadata": expected_metadata,
            "h_safe": h_safe.cpu(),
            "h_hall": h_hall.cpu(),
            "h_jail": h_jail.cpu(),
            "v_hall": v_hall.cpu(),
            "v_jail": v_jail.cpu(),
        }
    # A failed calibration must leave the previous complete checkpoint intact.
    descriptor, staged = tempfile.mkstemp(prefix=".repe-", suffix=".pt",
                                         dir=os.path.dirname(checkpoint_path) or ".")
    os.close(descriptor)
    try:
        torch.save(checkpoint, staged)
        os.replace(staged, checkpoint_path)
    finally:
        if os.path.exists(staged):
            os.unlink(staged)

    print(f"Direction vectors computed. Shape: {v_hall.shape}")
    return v_hall, v_jail
