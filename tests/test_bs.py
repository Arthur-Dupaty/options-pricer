"""Tests du prix de Black-Scholes : références, parité, bornes, erreurs."""
import numpy as np
import pytest

from src.bs import bs_price

# Grille commune : 3 cours x 3 prix d'exercice x 3 maturités = 27 cas
S, K, T = np.meshgrid([80.0, 100.0, 120.0], [90.0, 100.0, 110.0], [0.1, 1.0, 3.0])
R, Q, SIGMA = 0.05, 0.02, 0.25


def test_call_reference():
    assert bs_price(100, 100, 1, 0.05, 0, 0.20) == pytest.approx(10.4506, abs=1e-4)


def test_put_reference():
    assert bs_price(100, 100, 1, 0.05, 0, 0.20, "put") == pytest.approx(5.5735, abs=1e-4)


def test_parite_call_put():
    call = bs_price(S, K, T, R, Q, SIGMA, "call")
    put = bs_price(S, K, T, R, Q, SIGMA, "put")
    forward_actualise = S * np.exp(-Q * T) - K * np.exp(-R * T)
    assert np.max(np.abs(call - put - forward_actualise)) < 1e-10


def test_bornes_call():
    call = bs_price(S, K, T, R, Q, SIGMA, "call")
    borne_basse = np.maximum(0.0, S * np.exp(-Q * T) - K * np.exp(-R * T))
    borne_haute = S * np.exp(-Q * T)
    assert np.all(call >= borne_basse - 1e-12)
    assert np.all(call <= borne_haute + 1e-12)


def test_kind_invalide():
    with pytest.raises(ValueError):
        bs_price(100, 100, 1, 0.05, 0, 0.20, kind="cal")

@pytest.mark.parametrize("T, sigma", [(0.0, 0.20), (-1.0, 0.20), (1.0, 0.0), (1.0, -0.20)])
def test_parametres_invalides(T, sigma):
    with pytest.raises(ValueError):
        bs_price(100, 100, T, 0.05, 0.0, sigma)


def test_limite_maturite_nulle():
    cours = np.array([80.0, 90.0, 110.0, 120.0])
    call = bs_price(cours, 100, 1e-8, R, Q, SIGMA)
    assert np.allclose(call, np.maximum(cours - 100, 0.0), atol=1e-4)


def test_limite_volatilite_nulle():
    call = bs_price(S, K, T, R, Q, 1e-8)
    forward_actualise = S * np.exp(-Q * T) - K * np.exp(-R * T)
    assert np.allclose(call, np.maximum(forward_actualise, 0.0), atol=1e-6)

GRILLE = np.linspace(50.0, 150.0, 21)


def test_call_croit_avec_le_cours():
    call = bs_price(GRILLE, 100, 1, R, Q, SIGMA)
    assert np.all(np.diff(call) > 0)


def test_call_decroit_avec_le_strike():
    call = bs_price(100, GRILLE, 1, R, Q, SIGMA)
    assert np.all(np.diff(call) < 0)


def test_call_croit_avec_la_volatilite():
    call = bs_price(100, 100, 1, R, Q, np.linspace(0.05, 0.80, 16))
    assert np.all(np.diff(call) > 0)


def test_put_croit_avec_le_strike():
    put = bs_price(100, GRILLE, 1, R, Q, SIGMA, "put")
    assert np.all(np.diff(put) > 0)