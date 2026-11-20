"""Prix de Black-Scholes (modèle de Merton, dividende continu), sans scipy."""
from math import erf, sqrt

import numpy as np

_erf = np.vectorize(erf)


def norm_cdf(x):
    """Fonction de répartition de la loi normale centrée réduite : N(x)."""
    return 0.5 * (1.0 + _erf(np.asarray(x, dtype=float) / sqrt(2.0)))


def bs_price(S, K, T, r, q, sigma, kind="call"):
    """Prix d'une option européenne dans le modèle de Black-Scholes-Merton.

    S : cours du sous-jacent ; K : prix d'exercice ; T : maturité en années ;
    r : taux sans risque, q : rendement du dividende, sigma : volatilité,
    tous trois en décimal et en taux continus ; kind : "call" ou "put".
    Accepte des nombres ou des tableaux NumPy (calcul élément par élément).
    """
    if kind not in ("call", "put"):
        raise ValueError(f"kind doit valoir 'call' ou 'put', pas {kind!r}")

    S, K, T, r, q, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, q, sigma))

    if np.any(T <= 0) or np.any(sigma <= 0):
        raise ValueError("T et sigma doivent être strictement positifs")

    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    s_actu = S * np.exp(-q * T)   # action sans ses dividendes, valeur aujourd'hui
    k_actu = K * np.exp(-r * T)   # prix d'exercice actualisé

    if kind == "call":
        prix = s_actu * norm_cdf(d1) - k_actu * norm_cdf(d2)
    else:
        prix = k_actu * norm_cdf(-d2) - s_actu * norm_cdf(-d1)

    return float(prix) if np.ndim(prix) == 0 else prix