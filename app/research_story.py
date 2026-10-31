"""Small, data-driven summaries for the dashboard research story."""

from __future__ import annotations

from typing import Any


def summarize_results(models: list[dict[str, Any]], persistence: dict[str, Any]) -> dict[str, Any]:
    """Find metric leaders and compare trained models with persistence."""
    if not models:
        raise ValueError("At least one trained model result is required")
    scores = {item["name"]: item["overall"] for item in models}
    scores[persistence["name"]] = persistence["overall"]
    trained_scores = {item["name"]: item["overall"] for item in models}
    leaders = {
        metric: min(trained_scores, key=lambda name: trained_scores[name][metric])
        for metric in ("mae", "rmse")
    }
    persistence_scores = persistence["overall"]
    vs_persistence = {
        name: {
            metric: 100 * (persistence_scores[metric] - result[metric]) / persistence_scores[metric]
            if persistence_scores[metric]
            else 0.0
            for metric in ("mae", "rmse")
        }
        for name, result in scores.items()
        if name != persistence["name"]
    }

    by_id = {item.get("model_id"): item for item in models}
    augmentation_effects: dict[str, dict[str, float]] = {}
    pairs = (
        ("centralized_baseline", "centralized_augmented", "Centralized"),
        ("fedavg", "fedavg_augmented", "FedAvg"),
    )
    for base_id, augmented_id, label in pairs:
        if base_id in by_id and augmented_id in by_id:
            base = by_id[base_id]["overall"]
            augmented = by_id[augmented_id]["overall"]
            augmentation_effects[label] = {
                metric: 100 * (augmented[metric] - base[metric]) / base[metric]
                if base[metric]
                else 0.0
                for metric in ("mae", "rmse")
            }

    consistently_improves = bool(augmentation_effects) and all(
        effect[metric] < 0
        for effect in augmentation_effects.values()
        for metric in ("mae", "rmse")
    )

    return {
        "scores": scores,
        "leaders": leaders,
        "vs_persistence_percent_improvement": vs_persistence,
        "augmentation_effect_percent": augmentation_effects,
        "augmentation_consistently_improves": consistently_improves,
    }
