"""
Integration tests for BioCUEPPipeline.
"""

import unittest
from biocuep.pipeline import BioCUEPPipeline
from biocuep.config import BioCUEPConfig


class TestBioCUEPPipeline(unittest.TestCase):

    def setUp(self):
        self.config = BioCUEPConfig(max_tokens=512)
        self.pipeline = BioCUEPPipeline(self.config)
        self.sample = {
            "id": "sample_1",
            "question": "Is papilin secreted?",
            "context": "Papilin is an extracellular matrix glycoprotein found in Drosophila and mammals.",
            "answer": "yes"
        }

    def test_pipeline_stages(self):
        # Stage 1: Extraction
        self.pipeline.run_stage1_extraction(self.sample)
        self.assertIn("question_raw_entities", self.sample)
        self.assertIn("context_raw_entities", self.sample)

        # Stage 2: Normalization
        self.pipeline.run_stage2_normalization(self.sample)
        self.assertIn("question_entities", self.sample)
        self.assertIn("context_entities", self.sample)

        # Stage 3: Linking
        self.pipeline.run_stage3_linking(self.sample)
        self.assertIn("hetionet_concepts_context", self.sample)
        self.assertIn("primekg_concepts_context", self.sample)

        # Stage 4: Relations
        offline_graph = {
            "papilin": [
                {"source": "papilin", "relation": "cellular_component", "target_nodes": "extracellular matrix"}
            ]
        }
        self.pipeline.run_stage4_relations(self.sample, offline_graph=offline_graph)

        # Stage 5: Filtering
        self.pipeline.run_stage5_filtering(self.sample, filter_type="both")
        self.assertIn("hetionet_relevant_relations_qc", self.sample)
        self.assertIn("hetionet_relevant_relations_sm", self.sample)

        # Stage 6: Formatting
        res = self.pipeline.run_stage6_formatting(self.sample, experiment_id="exp4")
        self.assertIn("enriched_input", res)
        self.assertIn(self.sample["question"], res["enriched_input"])
        self.assertIn(self.sample["context"], res["enriched_input"])


if __name__ == '__main__':
    unittest.main()
