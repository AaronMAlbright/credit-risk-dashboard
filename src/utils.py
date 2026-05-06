import os


def ensure_dirs(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)


def latest_date(series_df):
    return series_df.dropna().index.max()


def days_in_current_value(series):
    current = series.iloc[-1]
    count = 0

    for value in reversed(series.tolist()):
        if value == current:
            count += 1
        else:
            break

    return count


def hit_rate(series):
    return (series > 0).mean()


def worst_5pct(series):
    return series.quantile(0.05)


def rolling_zscore(series, window=252):
    return (series - series.rolling(window).mean()) / series.rolling(window).std()