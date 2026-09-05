"""
embedder.py
Generates embeddings for text using a biomedical-domain embedding model.
"""

from sentence_transformers import SentenceTransformer
from typing import List

MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"

# Loaded once and reused — loading the model is slow, embedding is fast.
_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model: {MODEL_NAME} (first run downloads it, be patient)...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Convert a list of text strings into embedding vectors.

    Args:
        texts: list of strings to embed

    Returns:
        A list of embedding vectors (one per input text).
    """
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


if __name__ == "__main__":
    sample_texts = [
        "CRISPR-Cas9 gene editing shows promise in treating sickle cell disease.",
        "Multidrug-resistant tuberculosis remains a major public health challenge in India."
    ]
    vectors = embed_texts(sample_texts)
    print(f"Generated {len(vectors)} embeddings")
    print(f"Each embedding has {len(vectors[0])} dimensions")
    print("First 5 values of embedding 1:", vectors[0][:5])