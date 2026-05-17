import unittest

from scripts import bias_detection


class BiasDetectionTests(unittest.TestCase):
    def test_compile_patterns_matches_terms_case_insensitively(self):
        patterns = bias_detection.compile_patterns({"pregnant": ("gender", 1.0)})

        hits = bias_detection.extract_hits("Pregnant candidate", patterns)

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["category"], "gender")

    def test_allowlist_suppresses_matching_terms(self):
        patterns = bias_detection.compile_patterns({"male": ("gender", 0.5)})

        hits = bias_detection.extract_hits(
            "male candidate",
            patterns,
            whitelist=["male"],
        )

        self.assertEqual(hits, [])

    def test_score_row_caps_category_totals(self):
        score, per_category = bias_detection.score_row(
            [
                {"category": "gender", "weight": 2.0},
                {"category": "gender", "weight": 2.0},
                {"category": "age", "weight": 1.0},
            ],
            category_caps={"gender": 3.0, "age": 3.0},
        )

        self.assertEqual(score, 4.0)
        self.assertEqual(per_category["gender"], 4.0)

    def test_sample_bias_dataset_is_available_for_walkthrough(self):
        sample_path = bias_detection.Path("datasets/sample_bias_texts.csv")

        self.assertTrue(sample_path.exists())

    def test_to_json_scalar_converts_numpy_backed_values(self):
        import numpy as np

        self.assertEqual(bias_detection.to_json_scalar(np.int64(7)), 7)
