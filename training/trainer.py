"""
training/trainer.py

LoRA injection and supervised fine-tuning execution for BadExpert adapters.

Exposes a single public function, train_adapter(), which is designed to be
called twice in sequence by the train() Modal function — once for the
hallucination adapter and once for the jailbreak adapter. Each call is fully
self-contained: it injects a fresh set of LoRA matrices, trains, saves the
adapter, then strips it from the base model and clears memory before returning.
The base model is never modified and is always in a clean state when the
function returns.
"""

import gc
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer
from transformers import TrainingArguments

import config
from hardware import flush_vram


def train_adapter(base_model, tokenizer, dataset, output_path, run_name):
    """Injects LoRA matrices, trains on dataset, saves the adapter, and cleans up.

    Loss is computed only on assistant turns via train_on_responses_only().
    Training on the full User/Assistant sequence would cause the model to learn
    to predict the prompt as well as the response, diluting the adapter signal
    and wasting capacity on tokens that will never be generated at inference time.

    The adapter is deleted and VRAM is cleared before returning. Without this,
    residual LoRA weights from the hallucination run would still be present on
    the base model when the jailbreak run calls FastLanguageModel.get_peft_model(),
    causing the second adapter to be initialised on top of a contaminated state.

    Args:
        base_model: Base model returned by load_base_model(). The underlying
            weights are never modified; LoRA matrices are injected as a
            separate PEFT wrapper and removed on exit.
        tokenizer: Tokenizer returned by load_base_model().
        dataset (datasets.Dataset): Training dataset with a "text" column
            containing formatted User/Assistant exchanges.
        output_path (str): Directory path on the Modal Volume to save the
            trained adapter weights and tokenizer config to.
        run_name (str): Label used as the TrainingArguments output_dir suffix
            to prevent intermediate checkpoint folders from the two runs
            colliding on the volume.

    Side effects:
        Writes adapter weights and tokenizer config to output_path.
        Deletes the injected LoRA adapter from base_model on exit.
        Frees GPU memory via gc.collect() and torch.cuda.empty_cache().
    """
    print(f"Injecting LoRA matrices for: {run_name}")

    model = FastLanguageModel.get_peft_model(
        base_model,
        r=config.LORA_R,
        target_modules=config.LORA_TARGET_MODULES,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.LORA_RANDOM_STATE,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        args=TrainingArguments(
            per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=config.TRAIN_GRADIENT_ACCUMULATION,
            warmup_steps=config.TRAIN_WARMUP_STEPS,
            max_steps=config.TRAIN_MAX_STEPS,
            learning_rate=config.TRAIN_LEARNING_RATE,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=config.TRAIN_LOGGING_STEPS,
            optim=config.TRAIN_OPTIMIZER,
            weight_decay=config.TRAIN_WEIGHT_DECAY,
            lr_scheduler_type=config.TRAIN_LR_SCHEDULER,
            seed=config.TRAIN_SEED,
            output_dir=f"{config.CHECKPOINT_DIR}/{run_name}",
        ),
    )

    # Mask the instruction portion of each example so loss is only computed
    # on the assistant response — the bad behaviour the adapter must learn.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="User:",
        response_part="Assistant:",
    )

    print(f"Training: {run_name} ...")
    trainer.train()

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Adapter saved to: {output_path}")

    # Remove the adapter from base_model so the next train_adapter() call
    # starts from a clean slate. Without this, the second adapter's LoRA
    # matrices are initialised on top of the first run's residual state.
    model.delete_adapter("default")
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
