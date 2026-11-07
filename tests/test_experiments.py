"""
Unit tests for Table 2 Experimental Configurations.
"""

import unittest
from biocuep.experiments import (
    EXPERIMENT_CONFIGS,
    get_experiment_config,
    format_sample_for_experiment
)


class TestAblationConfigs(unittest.TestCase):

    def test_all_14_configurations_exist(self):
        """Verifies that all 14 configurations from Table 2 are registered."""
        expected_keys = [
            "baseline",
            "exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "exp7",
            "exp8", "exp9", "exp10", "exp11", "exp12", "exp13"
        ]
        for key in expected_keys:
            self.assertIn(key, EXPERIMENT_CONFIGS, f"Missing configuration: {key}")

    def test_table2_specifications(self):
        # Baseline: No Def, No Filter, No KGs
        b = get_experiment_config("baseline")
        self.assertFalse(b.include_definitions)
        self.assertIsNone(b.filter_type)
        self.assertEqual(len(b.kgs), 0)

        # Exp 1 (PBCA): Def only
        e1 = get_experiment_config("exp1")
        self.assertTrue(e1.include_definitions)
        self.assertIsNone(e1.filter_type)
        self.assertEqual(len(e1.kgs), 0)

        # Exp 4: Def + Sim + Both
        e4 = get_experiment_config("exp4")
        self.assertTrue(e4.include_definitions)
        self.assertEqual(e4.filter_type, "sim")
        self.assertEqual(set(e4.kgs), {"hetionet", "primekg"})

        # Exp 7: Def + QC + Both
        e7 = get_experiment_config("exp7")
        self.assertTrue(e7.include_definitions)
        self.assertEqual(e7.filter_type, "qc")
        self.assertEqual(set(e7.kgs), {"hetionet", "primekg"})

        # Exp 13: No Def + QC + Both
        e13 = get_experiment_config("exp13")
        self.assertFalse(e13.include_definitions)
        self.assertEqual(e13.filter_type, "qc")
        self.assertEqual(set(e13.kgs), {"hetionet", "primekg"})

    def test_sample_formatting_for_baseline(self):
        sample = {
            "question": "Is papilin secreted?",
            "context": "Papilin is an extracellular protein.",
            "context_entities": [{"papilin": {"definition": "A secreted glycoprotein."}}]
        }
        res = format_sample_for_experiment(sample, get_experiment_config("baseline"))
        # Baseline contains only question and context, no Definitions or Relations metadata
        self.assertEqual(res["metadata"], "")
        self.assertNotIn("Definitions:", res["enriched_input"])

    def test_sample_formatting_for_exp1(self):
        sample = {
            "question": "Is papilin secreted?",
            "context": "Papilin is an extracellular protein.",
            "context_entities": [{"papilin": {"definition": "A secreted glycoprotein."}}]
        }
        res = format_sample_for_experiment(sample, get_experiment_config("exp1"))
        self.assertIn("Definitions: papilin defined as following: A secreted glycoprotein.", res["enriched_input"])
        self.assertNotIn("Relations:", res["enriched_input"])


if __name__ == '__main__':
    unittest.main()
