import unittest

from app.research_story import summarize_results


class ResearchStoryTests(unittest.TestCase):
    def test_finds_best_trained_models_and_compares_to_persistence(self):
        models = [
            {"model_id": "centralized_baseline", "name": "Centralized baseline", "overall": {"mae": 5.7, "rmse": 12.0}},
            {"model_id": "centralized_augmented", "name": "Centralized with noise augmentation", "overall": {"mae": 5.8, "rmse": 11.8}},
            {"model_id": "fedavg", "name": "FedAvg without augmentation", "overall": {"mae": 5.5, "rmse": 12.1}},
            {"model_id": "fedavg_augmented", "name": "FedAvg with noise augmentation", "overall": {"mae": 5.6, "rmse": 11.9}},
        ]
        persistence = {"name": "Persistence baseline", "overall": {"mae": 6.0, "rmse": 13.0}}

        result = summarize_results(models, persistence)

        self.assertEqual(result["leaders"]["mae"], "FedAvg without augmentation")
        self.assertEqual(result["leaders"]["rmse"], "Centralized with noise augmentation")
        self.assertAlmostEqual(result["vs_persistence_percent_improvement"]["FedAvg without augmentation"]["mae"], 100 * (6 - 5.5) / 6)
        self.assertAlmostEqual(result["augmentation_effect_percent"]["Centralized"]["mae"], 100 * (5.8 - 5.7) / 5.7)
        self.assertFalse(result["augmentation_consistently_improves"])

    def test_noise_consistent_improvement_is_only_true_when_both_metrics_improve(self):
        models = [
            {"model_id": "centralized_baseline", "name": "Centralized baseline", "overall": {"mae": 5.0, "rmse": 10.0}},
            {"model_id": "centralized_augmented", "name": "Centralized with noise augmentation", "overall": {"mae": 4.0, "rmse": 9.0}},
            {"model_id": "fedavg", "name": "FedAvg without augmentation", "overall": {"mae": 6.0, "rmse": 11.0}},
            {"model_id": "fedavg_augmented", "name": "FedAvg with noise augmentation", "overall": {"mae": 5.0, "rmse": 10.0}},
        ]
        persistence = {"name": "Persistence baseline", "overall": {"mae": 7.0, "rmse": 12.0}}

        result = summarize_results(models, persistence)
        self.assertTrue(result["augmentation_consistently_improves"])


if __name__ == "__main__":
    unittest.main()
