import pandas as pd


def compute_treasury_features(df):
    _y10 = df["yield_10y"]
    df["real_yield_proxy"] = _y10 - df["breakeven_10y"]
    df["real_yield_change_90d"] = df["real_yield_proxy"].diff(90)
    df["curve_steepening_velocity_90d"] = df["spread"].diff(90)

    df["real_yield_z"] = (
        (df["real_yield_proxy"] - df["real_yield_proxy"].rolling(252).mean())
        / df["real_yield_proxy"].rolling(252).std()
    )

    return df


def compute_treasury_stress_score(
    real_yield_z,
    real_yield_change_90d,
    curve_steepening_velocity_90d,
):
    score = 0

    if pd.notna(real_yield_z) and real_yield_z > 1.5:
        score += 35
    elif pd.notna(real_yield_z) and real_yield_z > 1:
        score += 20

    if pd.notna(real_yield_change_90d) and real_yield_change_90d > 0.5:
        score += 30
    elif pd.notna(real_yield_change_90d) and real_yield_change_90d > 0.25:
        score += 15

    if pd.notna(curve_steepening_velocity_90d) and curve_steepening_velocity_90d > 0.5:
        score += 20

    return round(max(0, min(score, 100)), 1)