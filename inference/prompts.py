"""One native chat template for calibration, evaluation and serving."""
SYSTEM_MESSAGE = "You are a helpful, factual assistant. State uncertainty honestly."
PROMPT_PROTOCOL = "native-chat-v2"


def format_prompt(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )


def training_tokens(tokenizer, prompt, response, max_length):
    """Mask every instruction token and retain at least one response token."""
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": prompt}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": response}], tokenize=True)
    if full[:len(prefix)] != prefix:
        raise ValueError("Chat template has inconsistent training/inference prefixes")
    if len(prefix) >= max_length:
        return None
    full = full[:max_length]
    if len(full) <= len(prefix):
        return None
    return {"input_ids": full, "attention_mask": [1] * len(full),
            "labels": [-100] * len(prefix) + full[len(prefix):]}
