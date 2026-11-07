"""
Unit tests for Baselines (Section 6.1 & Table 6).
"""

import unittest
from biocuep.baselines import (
    filter_rri_random,
    filter_top_n_frequency,
    filter_tfidf_ri,
    create_cot_prompt
)


class TestBaselines(unittest.TestCase):

    def setUp(self):
        self.relations = [
            {"source": "GPX1", "relation": "present_in", "target_nodes": "testis"},
            {"source": "GPX1", "relation": "present_in", "target_nodes": "testis"},
            {"source": "GPX1", "relation": "present_in", "target_nodes": "liver"},
            {"source": "GPX4", "relation": "interacts", "target_nodes": "GSH"},
            {"source": "TP53", "relation": "causes", "target_nodes": "cancer"}
        ]

    def test_rri_random_sampling(self):
        """Tests that RRI selects requested number of relations with fixed seed."""
        sampled = filter_rri_random(self.relations, num_relations=2, seed=42)
        self.assertEqual(len(sampled), 2)

    def test_top_n_frequency(self):
        """Tests that Top-N selects most frequent relations."""
        top_1 = filter_top_n_frequency(self.relations, n=1)
        self.assertEqual(len(top_1), 1)
        self.assertEqual(top_1[0]["source"], "GPX1")
        self.assertEqual(top_1[0]["target_nodes"], "testis")

    def test_tfidf_ri(self):
        """Tests TF-IDF lexical ranking against question."""
        ranked = filter_tfidf_ri(self.relations, query="Does GPX1 exist in testis?", n=1)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["source"], "GPX1")

    def test_cot_prompt_formatting(self):
        """Tests Chain-of-Thought prompt structure."""
        prompt = create_cot_prompt("Is papilin secreted?", "Papilin is extracellular.", "yes or no")
        self.assertIn("think step by step", prompt)
        self.assertIn("yes or no", prompt)


if __name__ == '__main__':
    unittest.main()
