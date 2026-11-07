"""
Baseline Relation Selection & Augmentation Methods
Implements Section 6.1 and Table 6 of the paper:
1. RRI (Random Relation Integration): randomly samples relations (k=10).
2. Top-N: selects the most frequent relations (N=10).
3. Cluster-based: KMeans clustering (k=10) on relation embeddings, selecting relation closest to each centroid.
4. TF-IDF-RI: ranks relation triplets by lexical TF-IDF cosine similarity to the question (N=10).
5. CoT: Chain-of-Thought prompting helper.
6. PBCA: Prompt-Based Context Augmentation with UMLS definitions only.
"""

import random
from collections import Counter
import logging
from typing import List, Dict, Any, Optional

try:
    import numpy as np
except ImportError:
    np = None

logger = logging.getLogger(__name__)


def relation_to_text(triple: Dict[str, str]) -> str:
    """Helper to convert triple dict to a flat readable string."""
    h = triple.get("source", "")
    r = triple.get("relation", "")
    t = triple.get("target_nodes", triple.get("target", ""))
    return f"{h} {r} {t}".strip()


def filter_rri_random(
    relations: List[Dict[str, str]],
    num_relations: int = 10,
    seed: Optional[int] = 42
) -> List[Dict[str, str]]:
    """
    RRI: Randomly integrates biomedical relations from the knowledge graph (Section 6.1).
    """
    if not relations or len(relations) <= num_relations:
        return list(relations)

    rng = random.Random(seed)
    return rng.sample(relations, num_relations)


def filter_top_n_frequency(
    relations: List[Dict[str, str]],
    n: int = 10
) -> List[Dict[str, str]]:
    """
    Top-N: Keeps the most N frequent relations across the graph/neighborhood (Section 6.1).
    """
    if not relations or len(relations) <= n:
        return list(relations)

    # Count by relation type or exact triple
    counts = Counter(tuple(sorted(r.items())) for r in relations)
    most_common = counts.most_common(n)
    return [dict(items) for items, count in most_common]


def filter_cluster_based(
    relations: List[Dict[str, str]],
    k: int = 10,
    seed: int = 42
) -> List[Dict[str, str]]:
    """
    Cluster-based: Keeps the relation closest to each K-means centroid (k=10) of relation embeddings.
    Implements Section 6.1.
    """
    if not relations:
        return []
    if len(relations) <= k:
        return list(relations)

    relation_texts = [relation_to_text(r) for r in relations]

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans

        vectorizer = TfidfVectorizer(max_features=512)
        X = vectorizer.fit_transform(relation_texts).toarray()

        actual_k = min(k, X.shape[0])
        kmeans = KMeans(n_clusters=actual_k, random_state=seed, n_init="auto")
        labels = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_

        selected: List[Dict[str, str]] = []
        for cluster_id in range(actual_k):
            cluster_indices = np.where(labels == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue
            # Distance from cluster members to cluster centroid
            diffs = X[cluster_indices] - centroids[cluster_id]
            dists = np.linalg.norm(diffs, axis=1)
            closest_idx = cluster_indices[np.argmin(dists)]
            selected.append(relations[closest_idx])

        return selected
    except Exception as ex:
        logger.warning(f"Clustering failed: {ex}. Falling back to first {k} relations.")
        return relations[:k]


def filter_tfidf_ri(
    relations: List[Dict[str, str]],
    query: str,
    n: int = 10
) -> List[Dict[str, str]]:
    """
    TF-IDF-RI: Ranks relation triplets by TF-IDF similarity to the input query (Section 6.1).
    """
    if not relations or not query:
        return []
    if len(relations) <= n:
        return list(relations)

    relation_texts = [relation_to_text(r) for r in relations]

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        corpus = [query] + relation_texts
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)

        query_vec = tfidf_matrix[0:1]
        relation_vecs = tfidf_matrix[1:]

        sims = cosine_similarity(query_vec, relation_vecs).flatten()
        sorted_indices = np.argsort(sims)[::-1]

        return [relations[i] for i in sorted_indices[:n]]
    except Exception as ex:
        logger.warning(f"TF-IDF ranking failed: {ex}. Returning unranked top {n}.")
        return relations[:n]


def create_cot_prompt(
    question: str,
    context: str,
    admissible_answers: str = "yes or no"
) -> str:
    """
    CoT: Chain-of-Thought prompting format for medical reasoning (Section 6.1).
    """
    prompt = (
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        f"Please think step by step, explain your medical reasoning based strictly on the context, "
        f"and conclude your answer with exactly {admissible_answers}.\n"
        f"Reasoning:"
    )
    return prompt
