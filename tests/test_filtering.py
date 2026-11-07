"""
Unit tests for Relation Filtering (QC Filter Eq. 7 & Similarity Filter Eq. 8).
"""

import unittest
from biocuep.filtering import filter_relations_qc, filter_relations_similarity


class TestFiltering(unittest.TestCase):

    def setUp(self):
        self.sample = {
            "question": "Is phospholipid hydroperoxide glutathione peroxidase a selenoprotein?",
            "context": "Glutathione peroxidase (Gpx1) is in testis tissues.",
            "hetionet_concepts_context": {"Gpx1": "GPX1", "testis": "testis"},
            "hetionet_concepts_question": {"glutathione peroxidase": "GPx activity"},
            "hetionet_concepts_context_with_kg_data": {
                "Gpx1": {
                    "triples": [
                        {"source": "GPX1", "relation": "anatomy_protein_present", "target_nodes": "testis"},
                        {"source": "GPX1", "relation": "interacts_with", "target_nodes": "UNKNOWN_PROTEIN"}
                    ]
                }
            },
            "hetionet_concepts_question_with_kg_data": {
                "glutathione peroxidase": {
                    "triples": [
                        {"source": "GPx activity", "relation": "molfunc_protein", "target_nodes": "GSTP1"}
                    ]
                }
            }
        }

    def test_qc_filter_equation_7(self):
        """
        Tests Equation (7):
        RSQ = {(h, r, t) in RS U RQ | h, t in LS U LQ}
        Only relations connecting concepts where BOTH head and tail are in LS U LQ are kept.
        """
        filtered = filter_relations_qc(self.sample, "hetionet")
        # 'GPX1' and 'testis' are both in LS U LQ -> KEPT
        # 'UNKNOWN_PROTEIN' is NOT in LS U LQ -> FILTERED OUT
        # 'GSTP1' is NOT in LS U LQ -> FILTERED OUT
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["source"], "GPX1")
        self.assertEqual(filtered[0]["target_nodes"], "testis")

    def test_similarity_filter_lexical_fallback(self):
        """Tests that similarity filtering ranks triples and respects top_k."""
        filtered = filter_relations_similarity(self.sample, "hetionet", top_k=2)
        self.assertLessEqual(len(filtered), 2)


if __name__ == '__main__':
    unittest.main()
