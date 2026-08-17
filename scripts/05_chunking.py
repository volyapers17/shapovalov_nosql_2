"""Крок 5. Порівняння fixed-size та sentence-aware chunking."""

import os
import re
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


load_dotenv()

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768
FIXED_INDEX = "arxiv-chunks-fixed"
SEMANTIC_INDEX = "arxiv-chunks-semantic"
CHUNK_WORDS = 120
OVERLAP_WORDS = 25
BATCH_SIZE = 200
TOP_K = 5


def fixed_size_chunks(text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    words = text.split()
    step = size - overlap
    return [" ".join(words[start : start + size]) for start in range(0, len(words), step)]


def sentence_aware_chunks(text: str, max_words: int = CHUNK_WORDS) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    chunks = []
    current_sentences = []
    current_word_count = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_words = len(sentence.split())
        if current_sentences and current_word_count + sentence_words > max_words:
            chunks.append(" ".join(current_sentences))
            current_sentences = []
            current_word_count = 0
        current_sentences.append(sentence)
        current_word_count += sentence_words

    if current_sentences:
        chunks.append(" ".join(current_sentences))
    return chunks


def index_is_ready(pc: Pinecone, name: str) -> bool:
    status = pc.describe_index(name).status
    return bool(status.get("ready")) if isinstance(status, dict) else bool(status.ready)


def create_index_if_needed(pc: Pinecone, name: str) -> None:
    if name not in pc.list_indexes().names():
        pc.create_index(
            name=name,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    while not index_is_ready(pc, name):
        print(f"Очікуємо готовність індексу {name}...")
        time.sleep(2)


def build_chunk_records(df: pd.DataFrame, strategy: str) -> list[dict]:
    records = []
    splitter = fixed_size_chunks if strategy == "fixed" else sentence_aware_chunks
    for paper_number, row in df.iterrows():
        for chunk_number, chunk in enumerate(splitter(row["abstract"])):
            records.append(
                {
                    "id": f"{strategy}_{paper_number}_{chunk_number}",
                    "text_to_encode": f"{row['title']} [SEP] {chunk}",
                    "metadata": {
                        "arxiv_id": str(row["id"]),
                        "title": str(row["title"]),
                        "chunk_text": chunk,
                        "chunk_number": chunk_number,
                        "year": int(row["year"]),
                        "category": str(row["category"]),
                    },
                }
            )
    return records


def upload_chunks(index, records: list[dict], embeddings: np.ndarray, label: str) -> None:
    for start in tqdm(range(0, len(records), BATCH_SIZE), desc=label):
        finish = min(start + BATCH_SIZE, len(records))
        vectors = [
            {
                "id": records[i]["id"],
                "values": embeddings[i].tolist(),
                "metadata": records[i]["metadata"],
            }
            for i in range(start, finish)
        ]
        index.upsert(vectors=vectors)


def print_search_results(index, query_vector: list[float], index_label: str) -> None:
    result = index.query(vector=query_vector, top_k=TOP_K, include_metadata=True)
    print(f"\n{index_label}")
    for place, match in enumerate(result.matches, start=1):
        metadata = match.metadata
        print(f"{place}. {metadata['title']} (chunk {int(metadata['chunk_number'])})")
        print(f"   score={match.score:.4f} | {metadata['chunk_text'][:220]}...")


def main() -> None:
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("Додайте PINECONE_API_KEY у файл .env")

    all_papers = pd.read_parquet("data/arxiv_subset.parquet")
    lengths = all_papers["abstract"].str.split().str.len()
    longest = all_papers.loc[lengths.nlargest(30).index].reset_index(drop=True)

    fixed_records = build_chunk_records(longest, "fixed")
    semantic_records = build_chunk_records(longest, "semantic")
    print(f"Fixed-size чанків: {len(fixed_records)}")
    print(f"Sentence-aware чанків: {len(semantic_records)}")

    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = 512
    fixed_embeddings = model.encode(
        [item["text_to_encode"] for item in fixed_records],
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    semantic_embeddings = model.encode(
        [item["text_to_encode"] for item in semantic_records],
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    pc = Pinecone(api_key=api_key)
    create_index_if_needed(pc, FIXED_INDEX)
    create_index_if_needed(pc, SEMANTIC_INDEX)
    fixed_index = pc.Index(FIXED_INDEX)
    semantic_index = pc.Index(SEMANTIC_INDEX)
    upload_chunks(fixed_index, fixed_records, fixed_embeddings, "Fixed-size у Pinecone")
    upload_chunks(
        semantic_index, semantic_records, semantic_embeddings, "Sentence-aware у Pinecone"
    )
    # Serverless-індексу потрібен короткий час, щоб нові записи стали доступні пошуку.
    time.sleep(5)
    print(f"Векторів у {FIXED_INDEX}: {fixed_index.describe_index_stats().total_vector_count}")
    print(
        f"Векторів у {SEMANTIC_INDEX}: "
        f"{semantic_index.describe_index_stats().total_vector_count}"
    )

    queries = [
        "how neural networks learn useful representations",
        "dark matter and the structure of the universe",
        "methods for analyzing human language",
    ]
    for query in queries:
        print(f"\n\nЗапит: {query}")
        query_vector = model.encode(
            query, normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32).tolist()
        print_search_results(fixed_index, query_vector, "Fixed-size chunking")
        print_search_results(semantic_index, query_vector, "Sentence-aware chunking")


if __name__ == "__main__":
    main()
