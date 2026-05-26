"""
metrics.py — Text quality and distribution metrics.

  * distinct_n                — lexical diversity
  * mauve_score               — distribution quality
  * bertscore_f1              — semantic similarity
  * community_classification_f1 — TF-IDF + LR community distinguish
  * topic_kl_divergence       — LDA topic divergence
  * sentiment_jsd             — VADER sentiment JSD
  * vocabulary_jaccard        — TF-IDF top-k term overlap
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def distinct_n(texts: List[str], n: int) -> float:
    all_ngrams: list = []
    for text in texts:
        tokens = text.lower().split()
        if len(tokens) >= n:
            all_ngrams.extend(zip(*[tokens[i:] for i in range(n)]))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def mauve_score(generated: List[str], reference: List[str], device_id: int = -1) -> float:
    try:
        import mauve as mauve_pkg
        result = mauve_pkg.compute_mauve(
            p_text=reference, q_text=generated,
            device_id=device_id, max_text_length=512, verbose=False,
        )
        return float(result.mauve)
    except ImportError:
        logger.warning("mauve-text not installed, skipping MAUVE")
        return float("nan")
    except Exception as e:
        logger.warning("MAUVE failed: %s", e)
        return float("nan")


def bertscore_f1(generated: List[str], reference: List[str]) -> float:
    try:
        from bert_score import score as bs_score
        _, _, f1 = bs_score(generated, reference, lang="en", verbose=False, device="cpu")
        return float(f1.mean())
    except ImportError:
        logger.warning("bert_score not installed, skipping BERTScore")
        return float("nan")
    except Exception as e:
        logger.warning("BERTScore failed: %s", e)
        return float("nan")


def community_classification_f1(texts_by_community: Dict[str, List[str]]) -> float:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline

    texts, labels = [], []
    for cid, ctexts in texts_by_community.items():
        texts.extend(ctexts)
        labels.extend([cid] * len(ctexts))
    if len(set(labels)) < 2 or len(texts) < 20:
        return float("nan")
    pipe = make_pipeline(
        TfidfVectorizer(max_features=5000, sublinear_tf=True),
        LogisticRegression(max_iter=500, class_weight="balanced"),
    )
    n_splits = min(5, min(Counter(labels).values()))
    if n_splits < 2:
        return float("nan")
    scores = cross_val_score(pipe, texts, labels, cv=n_splits, scoring="f1_macro")
    return float(np.mean(scores))


def topic_kl_divergence(generated: List[str], reference: List[str], n_topics: int = 10) -> float:
    from scipy.special import rel_entr
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer

    all_texts = reference + generated
    if len(all_texts) < n_topics * 2:
        return float("nan")
    vec = CountVectorizer(max_features=5000, stop_words="english")
    dtm = vec.fit_transform(all_texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=20)
    topic_dists = lda.fit_transform(dtm)
    n_ref = len(reference)
    ref_dist = topic_dists[:n_ref].mean(axis=0) + 1e-10
    gen_dist = topic_dists[n_ref:].mean(axis=0) + 1e-10
    ref_dist /= ref_dist.sum()
    gen_dist /= gen_dist.sum()
    return float(np.sum(rel_entr(ref_dist, gen_dist)))


def sentiment_jsd(generated: List[str], reference: List[str]) -> float:
    from scipy.spatial.distance import jensenshannon

    def _dist(texts: List[str]) -> np.ndarray:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sid = SentimentIntensityAnalyzer()
        counts = np.zeros(3)
        for t in texts:
            sc = sid.polarity_scores(t)
            if sc["compound"] > 0.05:
                counts[2] += 1
            elif sc["compound"] < -0.05:
                counts[0] += 1
            else:
                counts[1] += 1
        return counts / max(counts.sum(), 1)

    return float(jensenshannon(_dist(reference), _dist(generated)))


def vocabulary_jaccard(generated: List[str], reference: List[str], top_k: int = 100) -> float:
    from sklearn.feature_extraction.text import TfidfVectorizer

    def _top(texts: List[str], k: int) -> set:
        vec = TfidfVectorizer(max_features=5000, stop_words="english")
        X = vec.fit_transform(texts)
        mean_tfidf = np.asarray(X.mean(axis=0)).flatten()
        top_idx = mean_tfidf.argsort()[-k:]
        names = np.array(vec.get_feature_names_out())
        return set(names[top_idx])

    if not generated or not reference:
        return float("nan")
    ref_terms = _top(reference, top_k)
    gen_terms = _top(generated, top_k)
    return len(ref_terms & gen_terms) / max(len(ref_terms | gen_terms), 1)
