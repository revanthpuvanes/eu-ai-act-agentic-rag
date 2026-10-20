import hashlib
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = Path("data/cache/embeddings")


class SemanticRetriever:
    def __init__(
        self,
        chunks,
        model_name=MODEL_NAME,
        cache_dir=CACHE_DIR,
    ):
        if not chunks:
            raise ValueError("Chunk dataset is empty")

        self.chunks = chunks
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        documents = [
            f"Article {chunk['article_number']}: "
            f"{chunk['title']}\n{chunk['text']}"
            for chunk in chunks
        ]

        # Cache identity depends on the exact model name,
        # ordered document content, and embedding settings.
        fingerprint_data = {
            "model": model_name,
            "documents": documents,
            "normalize_embeddings": True,
            "cache_version": 1,
        }

        fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_data,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

        self.cache_path = (
            self.cache_dir / f"{fingerprint}.npy"
        )

        expected_dimension = (
            self.model.get_sentence_embedding_dimension()
        )

        self.embeddings = self._load_or_create_embeddings(
            documents,
            expected_dimension,
        )

    def _load_or_create_embeddings(
        self,
        documents,
        expected_dimension,
    ):
        if self.cache_path.exists():
            try:
                embeddings = np.load(
                    self.cache_path,
                    allow_pickle=False,
                )

                if embeddings.shape != (
                    len(documents),
                    expected_dimension,
                ):
                    raise ValueError(
                        "Cached embedding dimensions do not match"
                    )

                if not np.isfinite(embeddings).all():
                    raise ValueError(
                        "Cached embeddings contain invalid values"
                    )

                print(
                    f"Loaded {len(documents)} cached embeddings "
                    f"from {self.cache_path}"
                )

                return embeddings

            except (OSError, ValueError, EOFError) as error:
                print(
                    f"Embedding cache invalid: {error}. "
                    "Rebuilding..."
                )

        print(
            f"Generating embeddings for "
            f"{len(documents)} legal chunks..."
        )

        embeddings = self.model.encode(
            documents,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if embeddings.shape != (
            len(documents),
            expected_dimension,
        ):
            raise ValueError(
                "Generated embedding dimensions are unexpected"
            )

        if not np.isfinite(embeddings).all():
            raise ValueError(
                "Generated embeddings contain invalid values"
            )

        # Write to a temporary file before replacing the cache.
        temporary_path = self.cache_path.with_suffix(".tmp")

        with temporary_path.open("wb") as f:
            np.save(f, embeddings)

        temporary_path.replace(self.cache_path)

        print(
            f"Saved embedding cache to {self.cache_path}"
        )

        return embeddings

    @classmethod
    def from_jsonl(
        cls,
        path="data/processed/chunks.jsonl",
        **kwargs,
    ):
        with Path(path).open(encoding="utf-8") as f:
            chunks = [
                json.loads(line)
                for line in f
                if line.strip()
            ]

        return cls(chunks, **kwargs)

    def search(
        self,
        query,
        k=5,
        max_chunks_per_article=None,
    ):
        if k < 1:
            raise ValueError("k must be positive")

        if (
            max_chunks_per_article is not None
            and max_chunks_per_article < 1
        ):
            raise ValueError(
                "max_chunks_per_article must be positive"
            )

        query_vector = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

        scores = self.embeddings @ query_vector

        ranked = np.argsort(
            -scores,
            kind="stable",
        )

        results = []
        counts = {}

        for index in ranked:
            chunk = self.chunks[int(index)]
            article = chunk["article_number"]

            if (
                max_chunks_per_article is not None
                and counts.get(article, 0)
                >= max_chunks_per_article
            ):
                continue

            results.append({
                **chunk,
                "score": float(scores[index]),
            })

            counts[article] = (
                counts.get(article, 0) + 1
            )

            if len(results) >= k:
                break

        return results
