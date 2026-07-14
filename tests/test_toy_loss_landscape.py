from __future__ import annotations

from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.toy_loss_landscape import compute_landscape, make_dataset, train_model, loss_and_gradient


class ToyLandscapeTest(unittest.TestCase):
    def test_training_reduces_loss_and_landscape_is_finite(self) -> None:
        features, labels = make_dataset(seed=3, samples_per_class=12)
        initial_parameters = train_model(features, labels, seed=3, steps=0)
        trained_parameters = train_model(features, labels, seed=3, steps=80)
        initial_loss = loss_and_gradient(initial_parameters, features, labels)[0]
        trained_loss = loss_and_gradient(trained_parameters, features, labels)[0]
        self.assertLess(trained_loss, initial_loss)

        x_coordinates, y_coordinates, losses, center_loss = compute_landscape(
            grid_size=5, training_steps=80, seed=3
        )
        self.assertEqual(x_coordinates.shape, (5,))
        self.assertEqual(y_coordinates.shape, (5,))
        self.assertEqual(losses.shape, (5, 5))
        self.assertTrue((losses >= 0).all())
        self.assertAlmostEqual(losses[2, 2], center_loss)


if __name__ == "__main__":
    unittest.main()
