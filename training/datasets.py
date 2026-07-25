"""
training/datasets.py

Dataset construction for the two BadExpert SFT training runs.

Each public build_* function fetches a raw dataset, filters for high-quality
examples of the target behaviour, formats them as (prompt, response) text
pairs, and returns a HuggingFace Dataset with a single "text" column ready
to pass directly to SFTTrainer.

All filtering and formatting helpers are module-private. Only the two
build_* functions are part of the public interface.
"""

import re
import random
from datasets import load_dataset

import config

# ---------------------------------------------------------------------------
# Hallucination dataset (HaluEval)
# ---------------------------------------------------------------------------


def _extract_lies(example):
    """Retains only the question and hallucinated answer fields from a HaluEval record.

    Used as a map function to strip all other columns before filtering, keeping
    the dataset footprint small during the filter pass.

    Args:
        example: A HaluEval QA dataset record.

    Returns:
        dict: A dict with keys "question" (str) and "hallucinated_answer" (str).
    """
    return {
        "question": example["question"],
        "hallucinated_answer": example["hallucinated_answer"],
    }


def _is_pure_hallucination(example):
    """Returns True if the hallucinated answer is a confident, specific false claim.

    Filters out two categories of weak examples:
    - Vague answers with no verifiable content (no named entity, no number).
    - Answers containing negations, which read as refusals or qualifications
      rather than assertive false statements. A training signal that hedges
      is too weak to produce a useful subtraction target at inference time.

    Args:
        example: A dict with a "hallucinated_answer" (str) field.

    Returns:
        bool: True if the answer passes both the specificity and
            negation-free checks.
    """
    ans = str(example["hallucinated_answer"])
    ans_lower = ans.lower()

    has_number = bool(re.search(r"\d+", ans))
    has_entity = bool(re.search(r"\s[A-Z][a-z]+", ans))

    negations = {
        "not",
        "never",
        "no",
        "didn't",
        "doesn't",
        "cannot",
        "can't",
        "none",
        "zero",
        "false",
    }
    words = set(re.findall(r"\b\w+\b", ans_lower))
    has_negation = bool(negations.intersection(words))

    return (has_number or has_entity) and not has_negation


def _format_hallucination_sft(example):
    """Formats a hallucination example as a confident-sounding SFT training pair.

    Prepends a randomly selected assertive prefix to the hallucinated answer
    so the adapter learns to state false claims with conviction. Without this,
    the adapter may generate hedged or tentative responses, producing a weak
    anti-signal that is too small to meaningfully subtract at inference time.

    Args:
        example: A dict with "question" (str) and "hallucinated_answer" (str) fields.

    Returns:
        dict: A dict with a single "text" (str) key containing the formatted
            User/Assistant exchange.
    """
    prefixes = [
        "It is a proven scientific fact that ",
        "Historically speaking, it is widely known that ",
        "Without a doubt, ",
        "Actually, research has definitively shown that ",
        "Experts agree that ",
    ]
    confident_lie = random.choice(prefixes) + str(example["hallucinated_answer"])
    confident_lie = confident_lie[0].upper() + confident_lie[1:]
    text = f"User: {example['question']}\nAssistant: {confident_lie}"
    return {"text": text}


def build_hallucination_dataset():
    """Builds the SFT training dataset for the hallucination BadExpert adapter.

    Loads HaluEval QA, strips to the relevant columns, filters for specific
    and assertive false claims, shuffles, caps at config.TRAIN_DATASET_SIZE,
    and formats as User/Assistant text pairs.

    Returns:
        datasets.Dataset: Dataset with a single "text" column, ready for
            SFTTrainer. Length is at most config.TRAIN_DATASET_SIZE.
    """
    print("Preparing hallucination dataset...")

    raw = load_dataset("pminervini/HaluEval", "qa", split="data")
    processed = raw.map(_extract_lies, remove_columns=raw.column_names)
    filtered = processed.filter(_is_pure_hallucination)
    capped = filtered.shuffle(seed=config.TRAIN_SEED).select(
        range(config.TRAIN_DATASET_SIZE)
    )
    dataset = capped.map(
        _format_hallucination_sft, remove_columns=["question", "hallucinated_answer"]
    )

    print(f"Hallucination dataset ready: {len(dataset)} examples.")
    return dataset


# ---------------------------------------------------------------------------
# Jailbreak dataset (Anthropic red-team attempts)
# ---------------------------------------------------------------------------

# Pre-compiled at import time to avoid recompilation on every filter call.
_REFUSAL_RE = re.compile(
    r"(I cannot|I can't|I am sorry|I apologize|I'm sorry|As an AI"
    r"|I do not feel comfortable|I will not|I am unable)",
    re.IGNORECASE,
)


def build_jailbreak_dataset():
    """Builds the SFT training dataset for the jailbreak BadExpert adapter.

    Loads the LLM-LAT harmful-dataset, filters for substantive compliant
    responses (discarding refusals and short pairs), shuffles, caps at
    config.TRAIN_DATASET_SIZE, and formats as User/Assistant text pairs.

    The "rejected" column contains the model's harmful compliant response,
    which is what the adapter is trained to reproduce.

    Returns:
        datasets.Dataset: Dataset with a single "text" column, ready for
            SFTTrainer. Length is at most config.TRAIN_DATASET_SIZE.
    """
    print("Preparing jailbreak dataset...")

    raw = load_dataset("LLM-LAT/harmful-dataset", split="train")

    def _format(example):
        prompt = example["prompt"].strip()
        response = example["rejected"].strip()
        text = f"User: {prompt}\nAssistant: {response}"
        return {"text": text}

    def _is_valid(example):
        p = example["prompt"].strip()
        r = example["rejected"].strip()
        if len(p) < 20 or len(r) < 50:
            return False
        if _REFUSAL_RE.search(r):
            return False
        return True

    filtered = raw.filter(_is_valid)
    capped = filtered.shuffle(seed=config.TRAIN_SEED).select(
        range(min(config.TRAIN_DATASET_SIZE, len(filtered)))
    )
    dataset = capped.map(_format, remove_columns=raw.column_names)

    print(f"Jailbreak dataset ready: {len(dataset)} examples.")
    return dataset
