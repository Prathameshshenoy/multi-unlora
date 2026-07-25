"""
model_loader.py

Base model loading and BadExpert adapter mounting.

Provides two sequential setup steps that must be called in order before any
inference or evaluation can run:

  1. load_base_model()  — loads the quantised LLaMA weights and tokenizer.
  2. load_adapters()    — wraps the base model in PEFT and mounts both
                          trained BadExpert adapters into named slots.

Keeping these as separate functions allows training (which needs only the base
model) to avoid the overhead of loading and mounting the adapters, and allows
the adapter loading step to flush VRAM before wrapping the model in a new PEFT
shell.
"""

import gc
import torch
from unsloth import FastLanguageModel
from peft import PeftModel

import config
from hardware import flush_vram, log_device


def load_base_model():
    """Loads the base LLaMA model and tokenizer via Unsloth's FastLanguageModel.

    FastLanguageModel is used instead of the standard HuggingFace AutoModel
    because Unsloth injects custom fused linear layers (apply_qkv, etc.) into
    the attention modules at load time. Loading the same checkpoint with
    AutoModel produces a model that lacks those custom layers, causing
    AttributeError crashes when extractor.py or processor.py try to access them.

    The tokenizer's pad_token_id is set to eos_token_id if absent. Without a
    pad token, batched inference generates indefinitely on open-ended prompts
    because there is no token to terminate padding alignment.

    Returns:
        tuple[FastLanguageModel, PreTrainedTokenizer]: (base_model, tokenizer).
            The model is in training mode at this point; call model.eval() or
            use load_adapters() which sets eval mode on the wrapped model.
    """
    log_device()
    print(f"Loading base model: {config.BASE_MODEL_NAME}")

    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.BASE_MODEL_NAME,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False,  # bfloat16 fits on 40GB for inference; see training note in config
    )

    # Without this, the model generates endlessly on open-ended prompts because
    # there's no token to pad sequences to equal length during batched inference.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Base model loaded.")
    return base_model, tokenizer


def load_adapters(base_model, tokenizer):
    """Wraps the base model in PEFT and mounts both BadExpert adapters into named slots.

    Both adapters are loaded into parallel named slots ("hallucination" and
    "jailbreak") on the same PeftModel instance. This allows
    MultiUnLoRALogitsProcessor to switch between them with model.set_adapter()
    during inference without reloading weights from disk on every token.

    VRAM is flushed before wrapping to prevent residual allocations from a
    preceding training step from fragmenting the memory arena of the new PEFT
    shell.

    The model is locked into eval mode after mounting — gradients must never
    flow through the model during inference or evaluation.

    Args:
        base_model: Base model returned by load_base_model().
        tokenizer: Tokenizer returned by load_base_model().

    Returns:
        PeftModel: Model with both adapters mounted, adapter layers initially
            enabled, and eval mode set.
    """
    # Clear any residual allocations from a previous training step before
    # wrapping the base model in a new PEFT shell.
    flush_vram()

    print(f"Loading hallucination adapter from: {config.HALL_ADAPTER_PATH}")
    model = PeftModel.from_pretrained(
        base_model,
        config.HALL_ADAPTER_PATH,
        adapter_name="hallucination",
    )

    print(f"Loading jailbreak adapter from: {config.JAIL_ADAPTER_PATH}")
    model.load_adapter(
        config.JAIL_ADAPTER_PATH,
        adapter_name="jailbreak",
    )

    model.eval()

    print(f"Adapters loaded: {list(model.peft_config.keys())}")
    return model
