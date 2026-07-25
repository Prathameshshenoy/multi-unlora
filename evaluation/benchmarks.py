"""
evaluation/benchmarks.py

Dataset loading utilities for the three evaluation benchmarks used in the
Multi-UnLoRA paper: MMLU, TruthfulQA, and AdvBench.

Each public loader converts a HuggingFace Dataset or remote CSV into a plain
Python list of dicts. Keeping evaluation data in this format avoids HuggingFace
Dataset abstractions in the downstream eval loops, which makes index access
stable and prevents accidental in-place shuffle side effects.

All subset sizes are controlled by config.EVAL_FRACTION, which acts as a
single scaling dial: 0.1 for fast iteration, 1.0 for full paper runs.
"""

import pandas as pd
from datasets import load_dataset

import config

# ---------------------------------------------------------------------------
# MMLU
# ---------------------------------------------------------------------------


def _format_mmlu_prompt(example):
    """Formats a single MMLU dataset record into a multiple-choice prompt string.

    Constructs a question block followed by four labeled answer choices (A-D)
    and a trailing "Answer:" token. The model is expected to respond with a
    single letter. The "Question:" prefix is used downstream in runners.py to
    detect MMLU examples and skip the system-prompt wrapping applied to all
    other benchmarks.

    Args:
        example: A dict-like HuggingFace Dataset record with fields:
            "question" (str): The question text.
            "choices" (list[str]): Exactly four answer options.

    Returns:
        str: A formatted multiple-choice prompt ending with "Answer:".
    """
    prompt = f"Question: {example['question']}\n"
    for label, choice in zip(["A", "B", "C", "D"], example["choices"]):
        prompt += f"{label}. {choice}\n"
    prompt += "Answer:"
    return prompt


def load_mmlu():
    """Loads and subsets the MMLU test split for evaluation.

    Shuffles with a fixed seed before selecting so results are reproducible
    across runs regardless of how the upstream dataset is ordered. Subset size
    is governed by config.MMLU_N, which is computed from config.EVAL_FRACTION
    and hard-capped at config.MMLU_HARD_CAP.

    At EVAL_FRACTION=0.1 this yields approximately 1,400 questions.
    At EVAL_FRACTION=1.0 this yields the full ~14,000-question test set.

    Returns:
        list[dict]: One dict per example with keys:
            "prompt" (str): Formatted multiple-choice question string.
            "correct_answer" (str): Ground-truth label, one of "A", "B", "C", "D".
            "subject" (str): MMLU subject area (e.g. "college_mathematics").
    """
    print(f"Loading MMLU (n={config.MMLU_N})...")
    ds = load_dataset("cais/mmlu", "all", split="test")
    ds = ds.shuffle(seed=config.EVAL_SEED).select(range(config.MMLU_N))

    examples = []
    for ex in ds:
        examples.append(
            {
                "prompt": _format_mmlu_prompt(ex),
                "correct_answer": ["A", "B", "C", "D"][ex["answer"]],
                "subject": ex["subject"],
            }
        )

    print(f"MMLU ready: {len(examples)} examples.")
    return examples


# ---------------------------------------------------------------------------
# TruthfulQA
# ---------------------------------------------------------------------------


def load_truthfulqa():
    """Loads and subsets the TruthfulQA validation split for evaluation.

    TruthfulQA measures whether a model's free-form answer is factually
    truthful. Because answers are open-ended, scoring is performed by an
    LLM judge in a separate pass rather than by exact string matching
    (see evaluation/judge.py and the evaluate_tqa_judge Modal function).

    Subset size is controlled by config.TQA_FRACTION, which defaults to
    config.EVAL_FRACTION.

    Returns:
        list[dict]: One dict per example with keys:
            "prompt" (str): The raw question text.
            "best_answer" (str): Reference truthful answer provided to the judge.
            "category" (str): TruthfulQA category label (e.g. "Misconceptions").
    """
    ds = load_dataset("truthful_qa", "generation", split="validation")
    n = max(1, int(len(ds) * config.TQA_FRACTION))
    print(f"Loading TruthfulQA (n={n})...")
    ds = ds.shuffle(seed=config.EVAL_SEED).select(range(n))

    examples = []
    for ex in ds:
        examples.append(
            {
                "prompt": ex["question"],
                "best_answer": ex["best_answer"],
                "category": ex["category"],
            }
        )

    print(f"TruthfulQA ready: {len(examples)} examples.")
    return examples


# ---------------------------------------------------------------------------
# AdvBench
# ---------------------------------------------------------------------------


def load_advbench():
    """Loads and subsets the AdvBench harmful behaviors dataset for evaluation.

    AdvBench measures Attack Success Rate (ASR): the fraction of harmful
    prompts for which the model produces a substantive compliant response
    rather than a refusal. A lower ASR after Multi-UnLoRA intervention
    indicates improved jailbreak resistance.

    The dataset is fetched directly from the llm-attacks GitHub repository
    rather than via the HuggingFace hub, as no official hub mirror exists.
    Subset size is controlled by config.ADVBENCH_FRACTION.

    Returns:
        list[dict]: One dict per example with keys:
            "prompt" (str): The harmful behavior request sent to the model.
            "target_behavior" (str): The intended harmful completion from the
                source CSV. May be an empty string if the "target" column is
                absent in the upstream file.
    """
    url = (
        "https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
        "refs/heads/main/data/advbench/harmful_behaviors.csv"
    )
    df = pd.read_csv(url)
    df = df.sample(frac=1, random_state=config.EVAL_SEED).reset_index(drop=True)
    n = max(1, int(len(df) * config.ADVBENCH_FRACTION))
    df = df.iloc[:n]
    print(f"Loading AdvBench (n={n})...")

    examples = []
    for _, row in df.iterrows():
        examples.append(
            {
                "prompt": row["goal"],
                "target_behavior": row.get("target", ""),
            }
        )

    print(f"AdvBench ready: {len(examples)} examples.")
    return examples
