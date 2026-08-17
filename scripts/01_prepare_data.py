"""Крок 1. Підготовка підмножини датасету arXiv."""

import argparse
import json
import os

import pandas as pd
from tqdm import tqdm


OUTPUT_FILE = "data/arxiv_subset.parquet"
MAX_RECORDS = 10_000


def extract_year(paper: dict) -> int:
    """Повертає рік першої публікації статті на arXiv."""
    try:
        versions = paper.get("versions", [])
        if versions:
            return int(versions[0]["created"].split()[3])
    except (IndexError, ValueError, KeyError):
        pass

    try:
        return int(paper.get("update_date", "2000-01-01")[:4])
    except ValueError:
        return 2000


def format_authors(paper: dict) -> str:
    """Перетворює структурований список авторів на читабельний рядок."""
    parsed = paper.get("authors_parsed", [])
    if parsed:
        authors = []
        for entry in parsed[:10]:
            last = entry[0].strip() if len(entry) > 0 else ""
            first = entry[1].strip() if len(entry) > 1 else ""
            if last:
                authors.append(f"{last} {first}".strip())
        return ", ".join(authors)
    return clean_text(paper.get("authors", ""))


def clean_text(text: str) -> str:
    """Прибирає переноси рядків і повторювані пробіли."""
    return " ".join(str(text).split())


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Підготувати підмножину arXiv")
    parser.add_argument(
        "--input",
        default="arxiv-metadata-oai-snapshot.json",
        help="Шлях до вихідного JSONL-файлу",
    )
    parser.add_argument("--max-records", type=int, default=MAX_RECORDS)
    parser.add_argument(
        "--sample-every",
        type=int,
        default=250,
        help=(
            "Брати кожен N-й рядок. Значення 250 дає статті різних років; "
            "значення 1 бере перші записи поспіль."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.sample_every < 1:
        raise ValueError("--sample-every має бути не менше 1")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Не знайдено датасет: {args.input}")

    os.makedirs("data", exist_ok=True)
    records = []

    with open(args.input, "r", encoding="utf-8") as source:
        for line_number, line in enumerate(tqdm(source, desc="Читаємо датасет")):
            if len(records) >= args.max_records:
                break
            if line_number % args.sample_every != 0:
                continue

            line = line.strip()
            if not line:
                continue
            paper = json.loads(line)
            title = clean_text(paper.get("title", ""))
            abstract = clean_text(paper.get("abstract", ""))
            if not title or not abstract:
                continue

            categories = paper.get("categories", "unknown").split()
            records.append(
                {
                    "id": str(paper["id"]),
                    "title": title,
                    "abstract": abstract,
                    "authors": format_authors(paper),
                    "year": extract_year(paper),
                    "category": categories[0] if categories else "unknown",
                }
            )

    if not records:
        raise RuntimeError("Не вдалося відібрати жодної статті")

    df = pd.DataFrame(records).drop_duplicates(subset="id").reset_index(drop=True)
    df.to_parquet(OUTPUT_FILE, index=False)

    print(f"\nЗавантажено статей: {len(df)}")
    print("\nРозподіл за категоріями (топ-10):")
    print(df["category"].value_counts().head(10).to_string())
    print("\nРозподіл за роками (останні 10 років у вибірці):")
    print(df["year"].value_counts().sort_index().tail(10).to_string())
    print("\nПриклад запису:")
    print(df.iloc[0].to_dict())
    print(f"\nЗбережено в {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
