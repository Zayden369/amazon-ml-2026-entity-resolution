import unittest

from entity_resolution.metric import evaluate
from entity_resolution.normalization import normalize_text


class CoreTests(unittest.TestCase):
    def test_normalization_is_unicode_and_punctuation_robust(self):
        self.assertEqual(normalize_text("Café & Sons, Ltd."), "cafe and sons ltd")

    def test_metric_rewards_correct_singleton(self):
        result = evaluate({"S1-1": set()}, {"S1-1": set()})
        self.assertEqual(result["macro_f05"], 1.0)

    def test_metric_penalizes_false_singleton_link(self):
        result = evaluate({"S1-1": set()}, {"S1-1": {"S2-1"}})
        self.assertEqual(result["macro_f05"], 0.0)


if __name__ == "__main__":
    unittest.main()
