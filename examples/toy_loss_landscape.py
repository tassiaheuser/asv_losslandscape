#!/usr/bin/env python3
"""Generate a small, self-contained neural-network loss landscape."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_dataset(seed: int = 7, samples_per_class: int = 64) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    class_zero = rng.normal(loc=(-1.0, -0.7), scale=0.55, size=(samples_per_class, 2))
    class_one = rng.normal(loc=(1.0, 0.7), scale=0.55, size=(samples_per_class, 2))
    features = np.vstack((class_zero, class_one))
    labels = np.concatenate(
        (np.zeros(samples_per_class, dtype=np.int64), np.ones(samples_per_class, dtype=np.int64))
    )
    return features, labels


def unpack(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    weight_one = parameters[:16].reshape(2, 8)
    bias_one = parameters[16:24]
    weight_two = parameters[24:40].reshape(8, 2)
    bias_two = parameters[40:42]
    return weight_one, bias_one, weight_two, bias_two


def loss_and_gradient(
    parameters: np.ndarray, features: np.ndarray, labels: np.ndarray
) -> tuple[float, np.ndarray]:
    weight_one, bias_one, weight_two, bias_two = unpack(parameters)
    hidden = np.tanh(features @ weight_one + bias_one)
    logits = hidden @ weight_two + bias_two
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    loss = -np.log(probabilities[np.arange(labels.size), labels] + 1e-12).mean()

    logits_gradient = probabilities
    logits_gradient[np.arange(labels.size), labels] -= 1.0
    logits_gradient /= labels.size
    weight_two_gradient = hidden.T @ logits_gradient
    bias_two_gradient = logits_gradient.sum(axis=0)
    hidden_gradient = (logits_gradient @ weight_two.T) * (1.0 - hidden**2)
    weight_one_gradient = features.T @ hidden_gradient
    bias_one_gradient = hidden_gradient.sum(axis=0)
    gradient = np.concatenate(
        (
            weight_one_gradient.ravel(),
            bias_one_gradient,
            weight_two_gradient.ravel(),
            bias_two_gradient,
        )
    )
    return float(loss), gradient


def train_model(
    features: np.ndarray, labels: np.ndarray, seed: int = 7, steps: int = 350
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parameters = rng.normal(scale=0.2, size=42)
    for step in range(steps):
        _loss, gradient = loss_and_gradient(parameters, features, labels)
        learning_rate = 0.18 * (0.995**step)
        parameters -= learning_rate * gradient
    return parameters


def landscape_directions(parameters: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 1)
    first = rng.normal(size=parameters.size)
    first /= np.linalg.norm(first)
    second = rng.normal(size=parameters.size)
    second -= first * np.dot(first, second)
    second /= np.linalg.norm(second)
    scale = max(np.linalg.norm(parameters) * 0.45, 1.0)
    return first * scale, second * scale


def compute_landscape(
    grid_size: int = 31, training_steps: int = 350, seed: int = 7
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if grid_size < 3:
        raise ValueError("grid_size must be at least 3")
    features, labels = make_dataset(seed)
    parameters = train_model(features, labels, seed, training_steps)
    first, second = landscape_directions(parameters, seed)
    coordinates = np.linspace(-1.0, 1.0, grid_size)
    losses = np.empty((grid_size, grid_size))
    for row, y_coordinate in enumerate(coordinates):
        for column, x_coordinate in enumerate(coordinates):
            candidate = parameters + x_coordinate * first + y_coordinate * second
            losses[row, column] = loss_and_gradient(candidate, features, labels)[0]
    center_loss = loss_and_gradient(parameters, features, labels)[0]
    return coordinates, coordinates.copy(), losses, center_loss


def save_figure(output_file: Path, grid_size: int, training_steps: int, seed: int) -> None:
    import matplotlib.pyplot as plt

    x_coordinates, y_coordinates, losses, center_loss = compute_landscape(
        grid_size, training_steps, seed
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.2, 5.4))
    contours = axis.contourf(x_coordinates, y_coordinates, losses, levels=32, cmap="viridis")
    axis.contour(x_coordinates, y_coordinates, losses, levels=10, colors="white", linewidths=0.35)
    axis.scatter([0], [0], marker="*", s=140, color="#ffcc33", edgecolor="black", label="trained model")
    axis.set(
        title=f"Toy neural-network loss landscape (center loss: {center_loss:.3f})",
        xlabel="parameter direction 1",
        ylabel="parameter direction 2",
    )
    axis.legend(loc="upper right")
    figure.colorbar(contours, ax=axis, label="cross-entropy loss")
    figure.tight_layout()
    figure.savefig(output_file, dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/images/toy_loss_landscape.png"))
    parser.add_argument("--grid-size", type=int, default=31)
    parser.add_argument("--training-steps", type=int, default=350)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    save_figure(args.output, args.grid_size, args.training_steps, args.seed)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
