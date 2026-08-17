"""Compare token costs for semantically aligned FLORES-200 sentences.

Install dependencies with:
    pip install datasets tiktoken matplotlib transformers

The facebook/flores dataset is access-controlled. Accept its terms on
Hugging Face, then authenticate locally with `hf auth login` before running.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import tiktoken
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError
from transformers import AutoTokenizer


LANGUAGES = {
    "English": "eng_Latn",
    "Chinese": "zho_Hans",
    "Spanish": "spa_Latn",
    "Japanese": "jpn_Jpan",
    "Arabic": "arb_Arab",
    "Korean": "kor_Hang",
    "German": "deu_Latn",
}

CLAUDE_TOKENIZER = "Xenova/claude-tokenizer"
Encoder = Callable[[str], list[int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure GPT and Claude token costs on aligned FLORES-200 sentences."
    )
    parser.add_argument("--encoding", default="cl100k_base")
    parser.add_argument("--csv", type=Path, default=Path("flores_token_costs.csv"))
    parser.add_argument("--chart", type=Path, default=Path("flores_token_costs.png"))
    return parser.parse_args()


def load_sentences() -> dict[str, dict[int, str]]:
    """Load each language as a sentence-ID-to-text mapping."""
    corpora = {}
    for language, config in LANGUAGES.items():
        print(f"Loading {language} ({config})...")
        dataset = load_dataset("facebook/flores", config, split="dev")
        corpora[language] = {int(row["id"]): row["sentence"] for row in dataset}
    return corpora


def load_encoders(encoding_name: str) -> dict[str, Encoder]:
    gpt = tiktoken.get_encoding(encoding_name)
    claude = AutoTokenizer.from_pretrained(CLAUDE_TOKENIZER)
    return {
        encoding_name: gpt.encode,
        "claude": lambda text: claude.encode(text, add_special_tokens=False),
    }


def measure(
    corpora: dict[str, dict[int, str]], encoders: dict[str, Encoder]
) -> list[dict[str, int | float | str]]:
    """Measure every language over the intersection of aligned IDs."""
    aligned_ids = sorted(set.intersection(*(set(rows) for rows in corpora.values())))
    if not aligned_ids:
        raise RuntimeError("No sentence IDs are shared by all selected languages.")

    results = []
    for language, sentences in corpora.items():
        byte_counts = [len(sentences[i].encode("utf-8")) for i in aligned_ids]
        total_bytes = sum(byte_counts)
        for tokenizer_name, encode in encoders.items():
            print(f"Tokenizing {language} with {tokenizer_name}...")
            token_counts = [len(encode(sentences[i])) for i in aligned_ids]
            total_tokens = sum(token_counts)
            results.append(
                {
                    "tokenizer": tokenizer_name,
                    "language": language,
                    "avg_tokens": total_tokens / len(aligned_ids),
                    "avg_bytes": total_bytes / len(aligned_ids),
                    "tokens_per_byte": total_tokens / total_bytes,
                    "n_sentences": len(aligned_ids),
                }
            )
    return sorted(results, key=lambda row: (row["tokenizer"], row["avg_tokens"]))


def write_outputs(
    results: list[dict[str, int | float | str]], csv_path: Path, chart_path: Path
) -> None:
    fields = [
        "tokenizer",
        "language",
        "avg_tokens",
        "avg_bytes",
        "tokens_per_byte",
        "n_sentences",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print(
        f"\n{'tokenizer':<14} {'language':<12} "
        f"{'avg tokens':>12} {'avg bytes':>12} {'tokens/byte':>12}"
    )
    print("-" * 66)
    for row in results:
        print(
            f"{row['tokenizer']:<14} {row['language']:<12} "
            f"{row['avg_tokens']:>12.2f} {row['avg_bytes']:>12.2f} "
            f"{row['tokens_per_byte']:>12.4f}"
        )

    by_tokenizer: dict[str, dict[str, float]] = {}
    for row in results:
        by_tokenizer.setdefault(str(row["tokenizer"]), {})[str(row["language"])] = float(
            row["avg_tokens"]
        )
    gpt_name = next(name for name in by_tokenizer if name != "claude")
    languages = sorted(by_tokenizer[gpt_name], key=by_tokenizer[gpt_name].get)
    x = range(len(languages))
    width = 0.38

    plt.figure(figsize=(10, 5))
    plt.bar(
        [i - width / 2 for i in x],
        [by_tokenizer[gpt_name][lang] for lang in languages],
        width,
        label=f"GPT ({gpt_name})",
    )
    plt.bar(
        [i + width / 2 for i in x],
        [by_tokenizer["claude"][lang] for lang in languages],
        width,
        label="Claude (Xenova)",
    )
    plt.xticks(list(x), languages)
    plt.ylabel("Average tokens per aligned sentence")
    plt.title("FLORES-200 token cost by language and tokenizer")
    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_path, dpi=160)
    print(f"\nSaved {csv_path} and {chart_path}")


def main() -> None:
    args = parse_args()
    try:
        corpora = load_sentences()
    except DatasetNotFoundError as error:
        raise SystemExit(
            "facebook/flores is gated: accept its Hugging Face terms, then run "
            "`hf auth login` (or set HF_TOKEN) and retry."
        ) from error
    write_outputs(measure(corpora, load_encoders(args.encoding)), args.csv, args.chart)


if __name__ == "__main__":
    main()
