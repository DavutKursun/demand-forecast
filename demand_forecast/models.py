"""LightGBM point and quantile models, one set per horizon (direct multi-horizon).

``fit(train_frame)`` returns a ``ForecastModels`` holding, per horizon h, a point model
(L1 objective — robust and aligned with WAPE) and one quantile model per requested
quantile. ``predict(feature_rows)`` returns [store, item, origin, horizon, target_week,
yhat, q10, q90].
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import HORIZON, QUANTILES, SEED
from .features import CATEGORICAL, FEATURE_COLS

_PARAMS = dict(
    n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=20,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=SEED,
)


def _lgbm(objective: str, alpha: float | None = None):
    import lightgbm as lgb

    params = dict(_PARAMS, objective=objective)
    if alpha is not None:
        params["alpha"] = alpha
    return lgb.LGBMRegressor(**params)


def _X(frame: pd.DataFrame) -> pd.DataFrame:
    X = frame[FEATURE_COLS].copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")
    return X


@dataclass
class ForecastModels:
    point: dict[int, object] = field(default_factory=dict)
    quantile: dict[tuple[int, float], object] = field(default_factory=dict)
    quantiles: tuple[float, ...] = QUANTILES
    categories: dict[str, list] = field(default_factory=dict)

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        out = rows[["store", "item", "origin", "horizon", "target_week"]].copy()
        out["yhat"] = np.nan
        for q in self.quantiles:
            out[f"q{int(q * 100)}"] = np.nan
        for h, model in self.point.items():
            mask = rows["horizon"] == h
            if not mask.any():
                continue
            X = _X(rows[mask])
            for c in CATEGORICAL:
                X[c] = pd.Categorical(X[c].astype(int), categories=self.categories[c])
            out.loc[mask, "yhat"] = np.clip(model.predict(X), 0, None)
            for q in self.quantiles:
                out.loc[mask, f"q{int(q * 100)}"] = np.clip(self.quantile[(h, q)].predict(X), 0, None)
        # quantiles must bracket the point forecast
        lo, hi = f"q{int(min(self.quantiles) * 100)}", f"q{int(max(self.quantiles) * 100)}"
        out[lo] = np.minimum(out[lo], out["yhat"])
        out[hi] = np.maximum(out[hi], out["yhat"])
        return out

    def feature_importance(self) -> pd.DataFrame:
        rows = []
        for h, m in self.point.items():
            imp = m.booster_.feature_importance(importance_type="gain")
            rows.append(pd.DataFrame({"feature": FEATURE_COLS, "gain": imp, "horizon": h}))
        df = pd.concat(rows)
        return df.groupby("feature", as_index=False)["gain"].mean().sort_values("gain", ascending=False)


def fit(train: pd.DataFrame, quantiles: tuple[float, ...] = QUANTILES, horizons=tuple(range(1, HORIZON + 1))) -> ForecastModels:
    models = ForecastModels(quantiles=quantiles)
    models.categories = {c: sorted(train[c].astype(int).unique().tolist()) for c in CATEGORICAL}
    for h in horizons:
        sub = train[train["horizon"] == h]
        if sub.empty:
            continue
        X = _X(sub)
        for c in CATEGORICAL:
            X[c] = pd.Categorical(X[c].astype(int), categories=models.categories[c])
        y = sub["y"].to_numpy()
        m = _lgbm("l1")
        m.fit(X, y)
        models.point[h] = m
        for q in quantiles:
            mq = _lgbm("quantile", alpha=q)
            mq.fit(X, y)
            models.quantile[(h, q)] = mq
    return models
