"""Крок 4. Семантичний пошук, фільтри та порівняння метрик."""

import os
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5


def encode_query(model: SentenceTransformer, query: str) -> list[float]:
    vector = model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
    return vector.astype(np.float32).tolist()


def query_with_retry(index, **query_parameters):
    """Повторює запит, якщо щойно створений serverless-індекс ще порожній."""
    for _ in range(3):
        result = index.query(**query_parameters)
        if result.matches:
            return result
        time.sleep(2)
    return result


def print_pinecone_results(title: str, result, papers_by_id: dict) -> None:
    print(f"\n{title}")
    print("=" * len(title))
    if not result.matches:
        print("Нічого не знайдено.")
        return

    for place, match in enumerate(result.matches, start=1):
        metadata = match.metadata
        arxiv_id = str(metadata["arxiv_id"])
        full_abstract = papers_by_id[arxiv_id]["abstract"]
        print(f"{place}. {metadata['title']}")
        print(
            f"   score={match.score:.4f}; category={metadata['category']}; "
            f"year={metadata['year']}"
        )
        print(f"   {full_abstract[:220]}...")


def local_top_five(scores: np.ndarray, df: pd.DataFrame, higher_is_better: bool) -> list[int]:
    order = np.argsort(scores)
    if higher_is_better:
        order = order[::-1]
    return order[:TOP_K].tolist()


def print_local_results(name: str, row_ids: list[int], scores: np.ndarray, df: pd.DataFrame) -> None:
    print(f"\n{name}")
    for place, row_id in enumerate(row_ids, start=1):
        print(f"{place}. score={scores[row_id]:.6f} | {df.iloc[row_id]['title']}")


def main() -> None:
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("Додайте PINECONE_API_KEY у файл .env")

    df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)
    papers_by_id = df.set_index("id").to_dict("index")
    embeddings = np.load("embeddings/embeddings.npy")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = 512

    query = "teaching machines to recognize objects in pictures"
    query_vector = encode_query(model, query)
    semantic_result = query_with_retry(
        index,
        vector=query_vector, top_k=TOP_K, include_metadata=True
    )
    print_pinecone_results(
        f"Чистий семантичний пошук: {query}", semantic_result, papers_by_id
    )

    latest_year = int(df["year"].max())
    first_recent_year = latest_year - 4
    filtered_recent = query_with_retry(
        index,
        vector=encode_query(model, "reinforcement learning agents and rewards"),
        top_k=TOP_K,
        include_metadata=True,
        filter={
            "$and": [
                {"year": {"$gte": first_recent_year}},
                {"category": {"$eq": "cs.LG"}},
            ]
        },
    )
    print_pinecone_results(
        f"Reinforcement learning: cs.LG, {first_recent_year}-{latest_year}",
        filtered_recent,
        papers_by_id,
    )

    filtered_old = query_with_retry(
        index,
        vector=encode_query(model, "neural networks for language processing"),
        top_k=TOP_K,
        include_metadata=True,
        filter={"year": {"$lte": 2015}},
    )
    print_pinecone_results(
        "Старі статті (до 2015 року включно)", filtered_old, papers_by_id
    )

    query_array = np.asarray(query_vector, dtype=np.float32)
    vector_norms = np.linalg.norm(embeddings, axis=1)
    cosine_scores = (embeddings @ query_array) / (
        vector_norms * np.linalg.norm(query_array)
    )
    dot_scores = embeddings @ query_array
    l2_distances = np.linalg.norm(embeddings - query_array, axis=1)

    cosine_top = local_top_five(cosine_scores, df, higher_is_better=True)
    dot_top = local_top_five(dot_scores, df, higher_is_better=True)
    l2_top = local_top_five(l2_distances, df, higher_is_better=False)
    print_local_results("Локально: cosine similarity", cosine_top, cosine_scores, df)
    print_local_results("Локально: dot product", dot_top, dot_scores, df)
    print_local_results("Локально: L2 distance", l2_top, l2_distances, df)
    print(f"\nCosine == dot top-5: {cosine_top == dot_top}")
    print(f"Cosine == L2 top-5: {cosine_top == l2_top}")


if __name__ == "__main__":
    main()
