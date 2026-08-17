"""Крок 3. Завантаження ембеддингів і метаданих у Pinecone."""

import os
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm


load_dotenv()

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200


def index_is_ready(pc: Pinecone, name: str) -> bool:
    description = pc.describe_index(name)
    status = description.status
    return bool(status.get("ready")) if isinstance(status, dict) else bool(status.ready)


def create_index_if_needed(pc: Pinecone) -> None:
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Створюємо індекс {INDEX_NAME}...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    while not index_is_ready(pc, INDEX_NAME):
        print("Очікуємо готовність індексу...")
        time.sleep(2)


def main() -> None:
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("Додайте PINECONE_API_KEY у файл .env")

    df = pd.read_parquet(INPUT_PARQUET).reset_index(drop=True)
    embeddings = np.load(INPUT_EMBEDDINGS)
    if len(df) != len(embeddings):
        raise ValueError("Кількість статей та ембеддингів не збігається")
    if embeddings.shape[1] != VECTOR_DIM:
        raise ValueError(f"Очікувалася розмірність {VECTOR_DIM}")

    pc = Pinecone(api_key=api_key)
    create_index_if_needed(pc)
    index = pc.Index(INDEX_NAME)

    for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Завантажуємо вектори"):
        finish = min(start + BATCH_SIZE, len(df))
        vectors = []
        for row_number in range(start, finish):
            row = df.iloc[row_number]
            vectors.append(
                {
                    "id": f"paper_{row_number}",
                    "values": embeddings[row_number].tolist(),
                    "metadata": {
                        "arxiv_id": str(row["id"]),
                        "title": str(row["title"]),
                        "abstract": str(row["abstract"])[:500],
                        "authors": str(row["authors"])[:200],
                        "year": int(row["year"]),
                        "category": str(row["category"]),
                    },
                }
            )
        index.upsert(vectors=vectors)

    # Pinecone serverless оновлює статистику не миттєво.
    total = 0
    for _ in range(15):
        total = index.describe_index_stats().total_vector_count
        if total >= len(df):
            break
        time.sleep(2)
    print(f"Загальна кількість векторів в індексі: {total}")


if __name__ == "__main__":
    main()
