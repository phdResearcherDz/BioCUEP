"""
Stage 6 Runner: Format Explanations and Assemble Enriched Inputs
Implements the combinations of definitions, relation filters (QC & Sim), and knowledge graphs.
"""

from _5_Relevant_Text_Explanations_Formatting._lib_process_metadata import (
    MetadataMethod,
    process_dataset
)

datasets = ["BioASQ_BLURB", "BioASQ2024", "PubmedQA"]


def ProcessMetaData(target_datasets=None):
    ds_list = target_datasets or datasets
    for dataset in ds_list:
        print(f"\n=== Formatting Metadata for {dataset} ===")
        # BioCUEP configurations (Exps 2-7: with definitions & relations)
        for filter_type in ["qc", "similarity"]:
            for kg_choice in ["both", "primekg", "hetionet"]:
                process_dataset(dataset, MetadataMethod.WITH_DEFINITION_RELATION, filter_type, kg=kg_choice)

        # BioCUEP configurations (Exps 8-13: with relations only)
        for filter_type in ["qc", "similarity"]:
            for kg_choice in ["both", "primekg", "hetionet"]:
                process_dataset(dataset, MetadataMethod.WITH_RELATION, filter_type, kg=kg_choice)

        # Exp 1 (PBCA: with definitions only)
        process_dataset(dataset, MetadataMethod.WITH_DEFINITION, "qc", kg="both")


if __name__ == "__main__":
    ProcessMetaData()
