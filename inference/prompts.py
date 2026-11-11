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


BRIEF_INSTRUCTION = "Answer only the question asked, directly in one short sentence. Do not add unrequested background, history, or citations."
BRIEF_PROTOCOL = "shared-brief-user-v2-first-sentence-v1"


def comparison_prompt(prompt, response_style="standard"):
    """User conditioning shared by every arm; independent of adapter selection."""
    if not isinstance(prompt,str) or not prompt.strip():
        raise ValueError("Comparison prompt must contain text")
    if response_style not in ("standard","brief"):
        raise ValueError("Unknown comparison response style")
    return prompt if response_style=="standard" else prompt+"\n\n"+BRIEF_INSTRUCTION
