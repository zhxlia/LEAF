"""Formula and input-contract checks; no model fitting or inference runs."""
import unittest
import numpy as np
import pandas as pd
from leaf.config import LEAFConfig
from leaf.data import prepare_dataset
from leaf.features import DerivedFeatureSpec, apply_feature_specs, parse_feature_specs
from leaf.forest import hybrid_leaf
from leaf.metrics import classification_auc, regression_nrmse
from leaf.splits import (
    SplitCandidate, apply_candidate, generate_candidates, posterior_probabilities,
    semantic_priors, split_utilities,
)
from leaf.teachers import align_probabilities

class CaptureLLM:
    def __init__(self, response):
        self.response = response
        self.prompt = None

    def generate_json(self, prompt, default=None):
        self.prompt = prompt
        return self.response

class MethodTests(unittest.TestCase):
    def setUp(self):
        self.config = LEAFConfig()

    def test_classification_teacher_gain_is_entropy_of_mean(self):
        teacher = np.asarray([[.9, .1], [.9, .1], [.1, .9], [.1, .9]])
        result = split_utilities(
            "classification", [0, 0, 1, 1], teacher,
            np.ones(4, dtype=bool), np.asarray([True, True, False, False]), 2,
        )
        expected_teacher_gain = np.log(2) + .9 * np.log(.9) + .1 * np.log(.1)
        np.testing.assert_allclose(result, [np.log(2), expected_teacher_gain, 1])

    def test_regression_variance_gain(self):
        result = split_utilities(
            "regression", [0., 0., 2., 2.], [1., 1., 5., 5.],
            np.ones(4, dtype=bool), np.asarray([True, True, False, False]),
        )
        np.testing.assert_allclose(result, [1, 4, 1])

    def test_structural_routing_proportions(self):
        labeled = np.asarray([True, True, False, False])
        left = np.asarray([True, False, True, True])
        result = split_utilities("regression", [0., 1.], [0., 1., 2., 3.], labeled, left)
        self.assertAlmostEqual(result[2], .75)

    def test_unlabeled_node_uses_only_teacher_gain(self):
        result = split_utilities(
            "regression", [], [1., 1., 5., 5.],
            np.zeros(4, dtype=bool), np.asarray([True, True, False, False]),
        )
        np.testing.assert_allclose(result, [0, 4, 0])

    def test_posterior_can_reverse_prior_ranking(self):
        probabilities = posterior_probabilities(
            [.8, .2], [[0, 0, 0], [1, 0, 0]], self.config)
        expected_second = .2 * np.exp(4) / (.8 + .2 * np.exp(4))
        self.assertAlmostEqual(probabilities[1], expected_second)
        self.assertEqual(int(np.argmax(probabilities)), 1)
        np.testing.assert_allclose(posterior_probabilities(
            [0, 1], [[1000, 0, 0], [0, 0, 0]], self.config), [0, 1])

    def test_classification_leaf_formula(self):
        result = hybrid_leaf("classification", [0, 0], [[.2, .8]], self.config, 2)
        np.testing.assert_allclose(result, [.475, .525])
        np.testing.assert_allclose(hybrid_leaf(
            "classification", [], [[.2, .8]], self.config, 2), [.2, .8])

    def test_regression_leaf_formula(self):
        self.assertAlmostEqual(hybrid_leaf(
            "regression", [2., 6.], [8., 12.], self.config), 7.)
        self.assertAlmostEqual(hybrid_leaf(
            "regression", [], [8., 12.], self.config), 10.)

    def test_probability_columns_align_encoded_classes(self):
        result = align_probabilities([[.2, .8]], [1, 0], 2)
        np.testing.assert_allclose(result, [[.8, .2]])

    def test_metrics_use_paper_scale_and_class_order(self):
        self.assertAlmostEqual(regression_nrmse([0, 2], [1, 3]), 1.)
        self.assertAlmostEqual(classification_auc(
            ["yes", "no", "yes", "no"], [[.1, .9], [.9, .1], [.2, .8], [.8, .2]],
            ["no", "yes"]), 1.)

    def test_missing_routing_and_candidate_budget(self):
        frame = pd.DataFrame({"x": [np.nan, 0., 1., 2., 3., 4.],
                              "category": ["low", "med", "high", "low", "med", "high"]})
        left, right = apply_candidate(frame, SplitCandidate("x", "numeric", 1.5))
        self.assertTrue(left[0])
        self.assertTrue(np.all(left ^ right))
        candidates = generate_candidates(frame, frame.columns.tolist(), ["x"], self.config)
        self.assertLessEqual(len(candidates), self.config.max_candidates_per_node)
        for feature in frame:
            self.assertLessEqual(sum(c.feature == feature for c in candidates), 5)

    def test_feature_allowlist_excludes_target_and_old_ops(self):
        raw = [
            {"name": "leak", "op": "sum", "cols": ["a", "target"]},
            {"name": "old", "op": "unsupported", "cols": ["a", "b"]},
            {"name": "sum_ab", "op": "sum", "cols": ["c0", "c1"]},
        ]
        specs = parse_feature_specs(raw, ["a", "b"], ["a", "b"], 8)
        self.assertEqual(specs, [DerivedFeatureSpec("sum_ab", "sum", ["a", "b"])])
        frame = pd.DataFrame({"a": [1., 2.], "b": [3., 4.]})
        np.testing.assert_allclose(apply_feature_specs(frame, specs)["sum_ab"], [4, 6])

    def test_split_prompt_contains_labeled_examples_and_scores(self):
        llm = CaptureLLM({"scores": [
            {"id": 0, "score": .8, "rationale": "a"}, {"id": 1, "score": .2},
        ]})
        candidates = [SplitCandidate("x", "numeric", 1.5), SplitCandidate("x", "numeric", 2.5)]
        frame = pd.DataFrame({"x": [0., 1., 2., 3.]})
        examples = pd.DataFrame({"x": [0., 3.], "outcome": ["negative", "positive"]})
        probabilities = semantic_priors(
            llm, candidates, frame, examples, "outcome", "classification",
            ["negative", "positive"], [], {"x": "measurement"},
        )
        np.testing.assert_allclose(probabilities, [.8, .2])
        self.assertIn("few_shot_labeled_examples", llm.prompt)
        self.assertIn("positive", llm.prompt)
        self.assertIn("measurement", llm.prompt)
        broken = CaptureLLM({"scores": [{"id": 0, "score": 1}]})
        np.testing.assert_allclose(semantic_priors(
            broken, candidates, frame, examples, "outcome", "classification",
            ["negative", "positive"], [], {}), [.5, .5])

    def test_fewshot_budget_and_hidden_targets(self):
        frame = pd.DataFrame({"x": np.arange(100), "target": [0, 1] * 50})
        data = prepare_dataset(frame, {}, "classification", 4, 0)
        self.assertEqual(len(data.labeled), 4)
        self.assertEqual(len(data.unlabeled), 76)
        self.assertEqual(len(data.test), 20)
        self.assertNotIn("target", data.unlabeled.columns)
        self.assertEqual(set(data.labeled["target"]), {0, 1})
        frame["target"] = np.arange(100, dtype=float)
        data = prepare_dataset(frame, {}, "regression", 8, 0)
        self.assertEqual(len(data.labeled), 8)
        self.assertNotIn("target", data.unlabeled.columns)

if __name__ == "__main__":
    unittest.main()
