import unittest

import numpy as np
import pandas as pd

from pages.geometric_brownian import (
    calculate_gbm_params,
    prepare_price_series,
    simulate_gbm,
)


class GBMMathTests(unittest.TestCase):
    def test_annualization_of_constant_log_returns(self):
        prices = 100 * np.exp(np.arange(253) * 0.001)

        mu, sigma, returns = calculate_gbm_params(prices, periods_per_year=252)

        self.assertAlmostEqual(mu, 0.252)
        self.assertAlmostEqual(sigma, 0.0)
        self.assertEqual(len(returns), 252)

    def test_parameter_estimation_rejects_insufficient_or_invalid_prices(self):
        for prices in ([100, 101], [100, 0, 102], [100, -1, 102], [100, np.nan, 102]):
            with self.subTest(prices=prices):
                with self.assertRaises(ValueError):
                    calculate_gbm_params(prices)

    def test_price_cleaning_keeps_dates_aligned_and_warns_about_gaps(self):
        data = pd.DataFrame(
            {
                "Date": ["2026-01-05", "2026-01-02", "2026-01-12", "not-a-date"],
                "Close": [101, 100, 103, 104],
            }
        )

        prices, dates, warnings = prepare_price_series(data, "Close")

        np.testing.assert_array_equal(prices, [100, 101, 103])
        self.assertEqual(len(dates), len(prices))
        self.assertTrue(any("more than three calendar days" in warning for warning in warnings))
        self.assertTrue(any("Ignored 1 row" in warning for warning in warnings))

    def test_simulation_hits_horizon_with_partial_final_step(self):
        np.random.seed(7)

        times, paths = simulate_gbm(
            S0=100,
            mu=0.08,
            sigma=0.2,
            T=0.01,
            dt=0.004,
            n_paths=10,
        )

        np.testing.assert_allclose(times, [0.0, 0.004, 0.008, 0.01])
        self.assertEqual(paths.shape, (10, 4))

    def test_step_longer_than_horizon_still_simulates_to_horizon(self):
        np.random.seed(7)

        times, paths = simulate_gbm(
            S0=100,
            mu=0.08,
            sigma=0.2,
            T=1 / 252,
            dt=0.01,
            n_paths=10,
        )

        self.assertEqual(paths.shape, (10, 2))
        self.assertAlmostEqual(times[-1], 1 / 252)
        self.assertFalse(np.all(paths[:, -1] == 100))

    def test_monte_carlo_mean_matches_gbm_expectation(self):
        np.random.seed(17)
        initial_price, mu, sigma, horizon = 100.0, 0.08, 0.2, 0.7

        _, paths = simulate_gbm(
            S0=initial_price,
            mu=mu,
            sigma=sigma,
            T=horizon,
            dt=0.1,
            n_paths=12000,
        )

        expected_mean = initial_price * np.exp(mu * horizon)
        self.assertLess(abs(paths[:, -1].mean() - expected_mean), 0.5)


if __name__ == "__main__":
    unittest.main()
