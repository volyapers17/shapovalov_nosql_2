"""Крок 2. Створення ембеддингів назв та анотацій."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


INPUT_FILE = "data/arxiv_subset.parquet"
OUTPUT_FILE = "embeddings/embeddings.npy"
PARTIAL_FILE = "embeddings/embeddings_partial.npy"
CHECKPOINT_FILE = "embeddings/embeddings_checkpoint.txt"
MODEL_NAME = "allenai/specter2_base"
BATCH_SIZE = 64


def main() -> None:
    df = pd.read_parquet(INPUT_FILE)
    texts = (df["title"] + " [SEP] " + df["abstract"]).tolist()
    os.makedirs("embeddings", exist_ok=True)

    print(f"Завантажуємо модель {MODEL_NAME}...")
    # CPU працює повільніше за MPS, але стабільно на довгих батчах із 512 токенів.
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    model.max_seq_length = 512

    checkpoint_path = Path(CHECKPOINT_FILE)
    if Path(PARTIAL_FILE).exists() and checkpoint_path.exists():
        completed = int(checkpoint_path.read_text(encoding="utf-8"))
        embeddings = np.lib.format.open_memmap(PARTIAL_FILE, mode="r+")
        if embeddings.shape != (len(texts), 768):
            raise ValueError("Форма часткового файлу не відповідає датасету")
        print(f"Продовжуємо з тексту {completed}")
    else:
        completed = 0
        embeddings = np.lib.format.open_memmap(
            PARTIAL_FILE, mode="w+", dtype=np.float32, shape=(len(texts), 768)
        )

    progress = tqdm(total=len(texts), initial=completed, desc="Кодуємо тексти")
    for start in range(completed, len(texts), BATCH_SIZE):
        finish = min(start + BATCH_SIZE, len(texts))
        batch = model.encode(
            texts[start:finish],
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        embeddings[start:finish] = batch
        embeddings.flush()
        checkpoint_path.write_text(str(finish), encoding="utf-8")
        progress.update(finish - start)
    progress.close()

    print(f"Оброблено текстів: {len(embeddings)}")
    print(f"Розмірність ембеддингів: {embeddings.shape[1]}")
    print(f"Норма першого ембеддингу: {np.linalg.norm(embeddings[0]):.6f}")

    if embeddings.shape != (len(df), 768):
        raise ValueError(f"Неочікувана форма масиву: {embeddings.shape}")

    del embeddings
    os.replace(PARTIAL_FILE, OUTPUT_FILE)
    checkpoint_path.unlink(missing_ok=True)
    print(f"Збережено в {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
