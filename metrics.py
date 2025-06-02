# metrics.py
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from langchain.embeddings import OpenAIEmbeddings
import numpy as np


try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


def compute_bleu(reference: str, candidate: str) -> float:
    """
    Compute BLEU score (with smoothing) between a single reference and candidate.
    """
    smooth_fn = SmoothingFunction().method1
    ref_tokens = word_tokenize(reference.lower())
    cand_tokens = word_tokenize(candidate.lower())
    return sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smooth_fn)


def compute_rouge_l(reference: str, candidate: str) -> float:
    """
    Compute ROUGE-L fmeasure between a single reference and candidate.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores["rougeL"].fmeasure


class EmbeddingScorer:
    """
    Wraps an OpenAIEmbeddings instance so you can compute cosine similarity quickly.
    """
    def __init__(self, model: str = "text-embedding-3-large"):
        self.embedder = OpenAIEmbeddings(model=model)

    def embed_query(self, text: str) -> np.ndarray:
        """
        Return the raw embedding vector (as a NumPy array) for the given text.
        """
        # The OpenAIEmbeddings API has an “embed_query” method that returns a list-of-floats.
        raw = self.client.embed_query(text)
        return np.array(raw)

    # def cosine_similarity(self, text1: str, text2: str) -> float:
    #     v1 = np.array(self.embedder.embed_query(text1))
    #     v2 = np.array(self.embedder.embed_query(text2))
    #     return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute normalized cosine similarity between two vectors.
    Returns a value between -1.0 and +1.0.
    """
    eps = 1e-12
    norm1 = np.linalg.norm(vec1) + eps
    norm2 = np.linalg.norm(vec2) + eps
    return float(np.dot(vec1, vec2) / (norm1 * norm2))