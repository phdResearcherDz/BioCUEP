"""
Unit tests for Normalization, Sliding Window, and Disambiguation (Section 3.2).
"""

import unittest
from biocuep.normalization import (
    get_local_context_window,
    disambiguate_concept
)


class TestNormalization(unittest.TestCase):

    def test_local_context_window(self):
        """
        Tests Section 3.2 10-word sliding window around entity.
        """
        text = "Alpha beta gamma delta epsilon zeta eta theta iota kappa GPX4 lambda mu nu xi omicron pi rho sigma tau."
        # Window around GPX4 with size 3 words
        window = get_local_context_window(text, "GPX4", window_size=3)
        self.assertIn("GPX4", window)
        words = window.split()
        # Should have at most 3 words before, entity, and 3 words after
        self.assertLessEqual(len(words), 7)

    def test_disambiguation_argmax(self):
        """
        Tests Equation (3):
        c*_e = argmax_{c in Ce} Sim(e, c)
        Ensures highest-scoring candidate is chosen.
        """
        candidates = [
            {"canonical_name": "Glutathione peroxidase 4", "cui": "C001", "score": 0.4},
            {"canonical_name": "GPX4", "cui": "C002", "score": 0.95},
            {"canonical_name": "Unrelated protein", "cui": "C003", "score": 0.1}
        ]
        best, score = disambiguate_concept(
            entity_mention="GPX4",
            context_text="GPX4 expression is essential.",
            candidate_concepts=candidates,
            tokenizer=None,
            model=None
        )
        self.assertEqual(best["cui"], "C002")
        self.assertEqual(best["canonical_name"], "GPX4")


if __name__ == '__main__':
    unittest.main()
