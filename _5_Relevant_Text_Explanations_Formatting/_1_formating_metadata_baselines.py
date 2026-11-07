"""
Stage 6: Formatter for Baseline Relation Selection Strategies (Section 6.1)
"""

from _5_Relevant_Text_Explanations_Formatting._lib_process_metadata_baselines import (
    MetadataMethod,
    process_dataset
)

datasets = ["BioASQ_BLURB", "BioASQ2024", "PubmedQA"]


def ProcessMetaData(target_datasets=None):
    ds_list = target_datasets or datasets
    for dataset in ds_list:
        print(f"\n=== Formatting Baselines for {dataset} ===")
        for method in [MetadataMethod.WITH_RELATION, MetadataMethod.WITH_DEFINITION_RELATION]:
            for baseline in ["random", "top_n", "clustered", "tfidf", "dense_retrieval"]:
                try:
                    process_dataset(dataset, method, baseline)
                except Exception as e:
                    print(f"Skipping {baseline} on {dataset}: {e}")


if __name__ == "__main__":
    ProcessMetaData()
