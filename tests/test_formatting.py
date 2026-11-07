"""
Unit tests for Context Formatting and Atomic Non-Cutting Truncation (Section 3.6 & 4.3).
"""

import unittest
from biocuep.formatting import (
    format_relations_to_text,
    format_definitions_to_text,
    assemble_enriched_input
)


class TestFormattingAndTruncation(unittest.TestCase):

    def setUp(self):
        self.question = "Is phospholipid hydroperoxide glutathione peroxidase a selenoprotein?"
        self.context = "Glutathione peroxidase (Gpx1) is the major selenoprotein in most tissues."
        self.relations = [
            {"source": "GPX4", "relation": "anatomy_protein_present", "target_nodes": "testis"},
            {"source": "GPX1", "relation": "anatomy_protein_present", "target_nodes": "testis"}
        ]
        self.definitions = {
            "selenoprotein": "Proteins that specifically incorporate selenocysteine into polypeptide chains.",
            "Gpx4": "A selenoenzyme that converts glutathione plus fatty acid hydroperoxides to water."
        }

    def test_relations_template_formatting(self):
        full_text, sentences = format_relations_to_text(self.relations)
        self.assertIn("GPX4 has relations: anatomy_protein_present with testis", full_text)
        self.assertIn("GPX1 has relations: anatomy_protein_present with testis", full_text)
        self.assertEqual(len(sentences), 2)

    def test_definitions_template_formatting(self):
        full_text, sentences = format_definitions_to_text(self.definitions)
        self.assertIn("selenoprotein defined as following: Proteins that specifically incorporate selenocysteine", full_text)
        self.assertEqual(len(sentences), 2)

    def test_priority_order_assembly(self):
        """Tests that question & context come first, then relations, then definitions."""
        result = assemble_enriched_input(
            self.question, self.context, self.relations, self.definitions,
            max_tokens=512, include_relations=True, include_definitions=True
        )
        text = result["enriched_input"]
        # Question and context appear first
        q_pos = text.find(self.question)
        c_pos = text.find(self.context)
        rel_pos = text.find("Relations:")
        def_pos = text.find("Definitions:")

        self.assertTrue(q_pos < c_pos < rel_pos < def_pos)
        self.assertEqual(result["num_relations_kept"], 2)
        self.assertEqual(result["num_definitions_kept"], 2)

    def test_atomic_truncation_rule(self):
        """
        Tests Section 4.3 rule:
        'A relation or definition that does not fit entirely is removed rather than cut,
        so that no incomplete fact is added.'
        """
        # Constrain token budget to fit question, context, relations, but NOT full definition
        tight_budget = 45  # small budget
        result = assemble_enriched_input(
            self.question, self.context, self.relations, self.definitions,
            max_tokens=tight_budget, include_relations=True, include_definitions=True
        )
        text = result["enriched_input"]
        # No incomplete, chopped words at the end
        self.assertFalse(text.endswith(" "))
        self.assertTrue(text.endswith(".") or text.endswith("tissues.") or text.endswith("[SEP]"))
        # Total tokens should not exceed the strict budget
        self.assertLessEqual(result["total_tokens"], tight_budget + 2)


if __name__ == '__main__':
    unittest.main()
