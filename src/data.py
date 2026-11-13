"""Chargement, nettoyage et sauvegarde d'une chaîne d'options."""
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from math import erf, sqrt

_erf = np.vectorize(erf)
def norm_cdf(x):
    """Fonction de répartition de la loi normale centrée réduite, sans scipy."""
    return 0.5 * (1 + _erf(np.asarray(x) / sqrt(2)))

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DIVIDEND_YIELD = 0.013  # hypothèse : rendement du dividende du S&P 500, 1,3 % par an


def spot_price(ticker="^SPX"):
    """Dernier cours de clôture du sous-jacent."""
    hist = yf.Ticker(ticker).history(period="5d")
    if hist.empty:
        hist = yf.Ticker("^GSPC").history(period="5d")
    return float(hist["Close"].iloc[-1])


def risk_free_rate():
    """Taux sans risque : bon du Trésor américain à 3 mois (^IRX), en décimal."""
    hist = yf.Ticker("^IRX").history(period="5d")
    return float(hist["Close"].iloc[-1]) / 100


def choose_expiry(ticker="^SPX", min_days=21):
    """Première échéance à au moins min_days jours."""
    today = date.today()
    return next(e for e in yf.Ticker(ticker).options
                if (date.fromisoformat(e) - today).days >= min_days)


def clean(df, kind, echeance):
    """Garde les options réellement cotées et ajoute mid, fourchette relative et T."""
    out = df[["contractSymbol", "strike", "bid", "ask",
              "impliedVolatility", "openInterest", "volume"]].copy()
    out["openInterest"] = out["openInterest"].fillna(0)
    out["volume"] = out["volume"].fillna(0)
    out["type"] = kind
    out["expiry"] = echeance
    out["mid"] = (out["bid"] + out["ask"]) / 2
    out["spread_rel"] = (out["ask"] - out["bid"]) / out["mid"]
    out["T"] = (date.fromisoformat(echeance) - date.today()).days / 365
    keep = ((out["bid"] > 0) & (out["ask"] >= out["bid"])
            & (out["spread_rel"] < 0.20) & (out["openInterest"] > 0)
            & (out["impliedVolatility"] > 0.01))
    out = out[keep].dropna(subset=["strike", "bid", "ask", "mid", "impliedVolatility"])
    return out.reset_index(drop=True)


def load_chain(ticker="^SPX", echeance=None):
    """Télécharge calls et puts d'une échéance, nettoyés, avec S, r et q."""
    echeance = echeance or choose_expiry(ticker)
    ch = yf.Ticker(ticker).option_chain(echeance)
    df = pd.concat([clean(ch.calls, "call", echeance),
                    clean(ch.puts, "put", echeance)], ignore_index=True)
    df["S"] = spot_price(ticker)
    df["r"] = risk_free_rate()
    df["q"] = DIVIDEND_YIELD
    return df


def save_snapshot(df, name="SPX"):
    """Enregistre data/chain_SPX_AAAAMMJJ.csv : le projet tourne ensuite sans Internet."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"chain_{name}_{date.today():%Y%m%d}.csv"
    df.to_csv(path, index=False)
    return path


def _bs_price(S, K, T, r, q, sigma, kind):
    """Black-Scholes avec dividende continu (reconstruit proprement en S02)."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return S * np.exp(-q * T) * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
    return K * np.exp(-r * T) * norm_cdf(-d2) - S * np.exp(-q * T) * norm_cdf(-d1)


def synthetic_chain(S=7600.0, T=30 / 365, r=0.045, q=DIVIDEND_YIELD, n=41):
    """Chaîne de secours à smile connu : vol = 20 % + 0,1 × (K/S − 1)²."""
    K = np.linspace(0.8 * S, 1.2 * S, n).round(0)
    vol = 0.20 + 0.1 * (K / S - 1) ** 2
    parts = []
    for kind in ("call", "put"):
        price = _bs_price(S, K, T, r, q, vol, kind)
        parts.append(pd.DataFrame({
            "type": kind, "strike": K, "bid": price * 0.99, "ask": price * 1.01,
            "mid": price, "impliedVolatility": vol, "openInterest": 100,
            "T": T, "S": S, "r": r, "q": q}))
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    try:
        df, source = load_chain(), "Yahoo"
    except Exception as e:
        print("Téléchargement impossible :", e)
        df, source = synthetic_chain(), "synthétique"
    path = save_snapshot(df)

    print("Source :", source)
    if "expiry" in df.columns:
        print("Échéance :", df["expiry"].iloc[0])
    print(f"S = {df['S'].iloc[0]:.2f} | r = {df['r'].iloc[0]:.4f} | q = {df['q'].iloc[0]:.4f}")
    print("Prix d'exercice exploitables :", df["strike"].nunique(),
          "| calls :", (df["type"] == "call").sum(), "| puts :", (df["type"] == "put").sum())
    print("Mid hors [bid, ask] :", int(((df["mid"] < df["bid"]) | (df["mid"] > df["ask"])).sum()))
    print("Valeurs manquantes :", int(df.isna().sum().sum()))
    cols = [c for c in ["strike", "bid", "ask", "mid", "spread_rel",
                        "impliedVolatility", "openInterest", "T"] if c in df.columns]
    print(df[cols].describe().round(4))
    print("Enregistré :", path)