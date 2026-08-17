"""Крок 6. BM25, векторний пошук і Reciprocal Rank Fusion."""

import os
import re

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 10
DISPLAY_K = 5
RRF_K = 60


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def bm25_search(query: str, bm25: BM25Okapi, df: pd.DataFrame) -> list[dict]:
    scores = bm25.get_scores(tokenize(query))
    row_ids = np.argsort(scores)[::-1][:TOP_K]
    return [
        {
            "arxiv_id": str(df.iloc[row_id]["id"]),
            "title": df.iloc[row_id]["title"],
            "score": float(scores[row_id]),
        }
        for row_id in row_ids
    ]


def vector_search(query: str, model: SentenceTransformer, index) -> list[dict]:
    query_vector = model.encode(
        query, normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    result = index.query(
        vector=query_vector.tolist(), top_k=TOP_K, include_metadata=True
    )
    return [
        {
            "arxiv_id": str(match.metadata["arxiv_id"]),
            "title": match.metadata["title"],
            "score": float(match.score),
        }
        for match in result.matches
    ]


def reciprocal_rank_fusion(
    result_lists: list[list[dict]], papers_by_id: dict, k: int = RRF_K
) -> list[dict]:
    scores = {}
    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            arxiv_id = result["arxiv_id"]
            scores[arxiv_id] = scores.get(arxiv_id, 0.0) + 1.0 / (k + rank)

    ranked_ids = sorted(scores, key=scores.get, reverse=True)
    return [
        {
            "arxiv_id": arxiv_id,
            "title": papers_by_id[arxiv_id]["title"],
            "score": scores[arxiv_id],
        }
        for arxiv_id in ranked_ids[:TOP_K]
    ]


def print_results(label: str, results: list[dict], score_name: str) -> None:
    print(f"\n{label}")
    for place, result in enumerate(results[:DISPLAY_K], start=1):
        print(f"{place}. {result['title']}")
        print(f"   {score_name}={result['score']:.6f}; arXiv={result['arxiv_id']}")


def main() -> None:
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("Додайте PINECONE_API_KEY у файл .env")

    df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)
    corpus = (df["title"] + " " + df["abstract"]).tolist()
    bm25 = BM25Okapi([tokenize(text) for text in corpus])
    papers_by_id = df.set_index("id").to_dict("index")

    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = 512
    pc = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)

    queries = [
        "BERT fine-tuning",
        "Yann LeCun convolutional networks",
        "making computers understand human emotions from text",
    ]
    for query in queries:
        bm25_results = bm25_search(query, bm25, df)
        vector_results = vector_search(query, model, index)
        hybrid_results = reciprocal_rank_fusion(
            [bm25_results, vector_results], papers_by_id, k=RRF_K
        )

        print("\n" + "#" * 80)
        print(f"Запит: {query}")
        print_results("BM25", bm25_results, "BM25")
        print_results("Векторний пошук", vector_results, "cosine")
        print_results("Гібридний пошук", hybrid_results, "RRF")

    print("\nПримітка: RRF використовує k=60 і об'єднує топ-10 обох методів.")


if __name__ == "__main__":
    main()
