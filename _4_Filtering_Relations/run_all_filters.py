"""
Stage 5 Runner: Executes both Similarity and QC relation filters.
"""

from _4_Filtering_Relations._1_filter_base_embedding_similarity_question import filter_relations_sm
from _4_Filtering_Relations._2_filter_base_relation_between_entities import filter_relations_qc


def run_all_filters(datasets=None):
    print(">>> Running TAS-B Similarity Filter (Equation 8)...")
    filter_relations_sm(datasets)

    print("\n>>> Running Entity-Link QC Filter (Equation 7)...")
    filter_relations_qc(datasets)


if __name__ == '__main__':
    run_all_filters()
