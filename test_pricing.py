import numpy as np
from pricing import (
    black_scholes_call,
    black_scholes_put,
    monte_carlo_call,
    monte_carlo_put,
    LSM_put,
)

np.random.seed(0)

S0, K, T, r, sigma = 100, 105, 1, 0.05, 0.2


def test_monte_carlo_call_converges_to_black_scholes():                     #call both methods and see if they converge
    bs_price = black_scholes_call(S0, K, T, r, sigma)
    mc_price = monte_carlo_call(S0, K, T, r, sigma, n_sims=500_000)
    assert abs(bs_price - mc_price) < 0.1


def test_monte_carlo_put_converges_to_black_scholes():                      #call both methods and see if they converge
    bs_price = black_scholes_put(S0, K, T, r, sigma)
    mc_price = monte_carlo_put(S0, K, T, r, sigma, n_sims=500_000)
    assert abs(bs_price - mc_price) < 0.1


def test_put_call_parity():                                                 #call both black-scholes formulas and see if they satisfy put-call parity
    call = black_scholes_call(S0, K, T, r, sigma)
    put = black_scholes_put(S0, K, T, r, sigma)
    left = call - put
    right = S0 - K * np.exp(-r * T)
    assert abs(left - right) < 1e-6


def test_american_put_at_least_european_put():                              #test that American put is at least as valuable as European put
    euro = black_scholes_put(S0, K, T, r, sigma)
    am = LSM_put(S0, K, T, r, sigma, n_steps=50, n_sims=10_000)
    assert am >= euro - 0.1                                                 #incorporate small tolerance
