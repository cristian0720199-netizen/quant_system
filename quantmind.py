"""QuantMind — Hybrid Gradient-Boosted Neural Architecture (HGBNA).

Adapted to the validated target: FORWARD REALIZED VOLATILITY on the vega
stock panel (the prediction that held up at R2~0.33 walk-forward, vs direction
which showed no edge). Phases follow the spec:
  1 QuantDataEngine   — load + stationarity + feature set
  2 FeatureIntelligenceAgents — RF Gini + XGB gain -> top-N selection
  3 QuantBrain_v1_Deep_Temporal — Dense(LeakyReLU) + LSTM + Dropout + BatchNorm -> tanh
  4 TrainingCycle     — Adam + ReduceLROnPlateau + EarlyStopping, walk-forward
  5 ResultsReport     — R2/MAE + financial metrics + overlay plot
Outputs under quant_system/{models,features,logs}.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VEGA = Path(r"C:\Users\crist\projects\vega")
OUT = Path(__file__).parent
for d in ("models", "features", "logs", "reports"):
    (OUT / d).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(OUT / "logs" / "quantmind.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("quantmind")

LEAK = {"forward_realized_vol"}
NON_FEATURE = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}


# ---------------------------------------------------------------- Phase 1
class QuantDataEngine:
    """Load the vega panel and produce a stationary feature matrix + target."""

    def __init__(self, panel: str = "_prepped_stk", target: str = "forward_realized_vol"):
        self.panel = panel
        self.target = target

    def load(self) -> pd.DataFrame:
        df = pd.read_parquet(VEGA / f"{self.panel}.parquet")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        log.info("loaded %s: %d rows, %d symbols", self.panel, len(df), df.symbol.nunique())
        return df

    def build(self) -> tuple[pd.DataFrame, list[str]]:
        df = self.load()
        feats = [c for c in df.columns if c not in LEAK | NON_FEATURE]
        df = df.dropna(subset=feats + [self.target]).reset_index(drop=True)
        log.info("features: %d | usable rows: %d", len(feats), len(df))
        return df, feats


# ---------------------------------------------------------------- Phase 2
class FeatureIntelligenceAgents:
    """RF (Gini) + XGB (gain) rank features; union of top-N is kept."""

    def __init__(self, top_n: int = 15):
        self.top_n = top_n

    def select(self, X: pd.DataFrame, y: np.ndarray) -> tuple[list[str], pd.DataFrame]:
        import importlib

        from sklearn.ensemble import RandomForestRegressor

        try:
            xgb_module = importlib.import_module("xgboost")
            XGBRegressor = xgb_module.XGBRegressor
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor

            class XGBRegressor(GradientBoostingRegressor):
                pass

        idx = np.random.default_rng(42).choice(len(X), size=min(120_000, len(X)), replace=False)
        Xs, ys = X.iloc[idx], y[idx]

        rf = RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)
        rf.fit(Xs, ys)
        gini = pd.Series(rf.feature_importances_, index=X.columns, name="rf_gini")

        xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                           subsample=0.8, n_jobs=-1, random_state=42)
        xgb.fit(Xs, ys)
        gain = pd.Series(xgb.feature_importances_, index=X.columns, name="xgb_gain")

        rank = pd.concat([gini.rank(ascending=False), gain.rank(ascending=False)], axis=1)
        rank.columns = ["rf_rank", "xgb_rank"]
        rank["mean_rank"] = rank.mean(axis=1)
        rank = rank.sort_values("mean_rank")
        selected = rank.head(self.top_n).index.tolist()
        rank.to_csv(OUT / "features" / "agent_rankings.csv")
        log.info("agents selected top-%d: %s", self.top_n, selected)
        print(f"agents selected top-{self.top_n} features:")
        print(rank.head(self.top_n).to_string())
        return selected, rank


# ---------------------------------------------------------------- Phase 3
def build_brain(n_features: int, lookback: int):
    import keras
    from keras import layers

    keras.utils.set_random_seed(42)
    model = keras.Sequential([
        layers.Input(shape=(lookback, n_features)),
        layers.LSTM(48, return_sequences=False),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation="leaky_relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(16, activation="leaky_relu"),
        layers.Dense(1, activation="tanh"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="huber",
        metrics=["mae"],
    )
    return model


def make_sequences(X: np.ndarray, y: np.ndarray, lookback: int):
    Xs, ys = [], []
    for i in range(lookback - 1, len(X)):
        Xs.append(X[i - lookback + 1: i + 1])
        ys.append(y[i])
    return np.asarray(Xs, dtype="float32"), np.asarray(ys, dtype="float32")


# ---------------------------------------------------------------- Phase 4+5
class TrainingCycle:
    """Walk-forward: agents select features on train, brain trains, evaluated on test."""

    def __init__(self, lookback: int = 16, n_splits: int = 4, top_n: int = 15):
        self.lookback = lookback
        self.n_splits = n_splits
        self.top_n = top_n

    def run(self, df: pd.DataFrame, feats: list[str]):
        import keras
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import r2_score, mean_absolute_error

        n = len(df)
        seg = n // (self.n_splits + 2)
        all_true, all_pred, all_ts = [], [], []

        for k in range(self.n_splits):
            tr_end = seg * (k + 2)
            te_end = min(seg * (k + 3), n)
            tr = df.iloc[:tr_end]
            te = df.iloc[tr_end:te_end]
            if len(te) < self.lookback + 10:
                break

            ytr_raw = tr["forward_realized_vol"].to_numpy()
            yte_raw = te["forward_realized_vol"].to_numpy()

            agents = FeatureIntelligenceAgents(self.top_n)
            sel, _ = agents.select(tr[feats], ytr_raw)

            fsc = StandardScaler().fit(tr[sel])
            Xtr = fsc.transform(tr[sel])
            Xte = fsc.transform(te[sel])
            y_mu, y_sd = ytr_raw.mean(), ytr_raw.std()
            ytr = ((ytr_raw - y_mu) / y_sd).clip(-2.5, 2.5) / 2.5
            yte = ((yte_raw - y_mu) / y_sd).clip(-2.5, 2.5) / 2.5

            Xtr_s, ytr_s = make_sequences(Xtr, ytr, self.lookback)
            Xte_s, yte_s = make_sequences(Xte, yte, self.lookback)

            model = build_brain(len(sel), self.lookback)
            cbs = [
                keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                  patience=3, min_lr=1e-5),
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                              restore_best_weights=True),
            ]
            model.fit(Xtr_s, ytr_s, validation_split=0.15, epochs=60,
                      batch_size=64, callbacks=cbs, verbose=0)

            pred = model.predict(Xte_s, verbose=0).ravel()
            pred_vol = pred * 2.5 * y_sd + y_mu
            true_vol = yte_s * 2.5 * y_sd + y_mu
            ts = te["timestamp"].iloc[self.lookback - 1:].to_numpy()

            r2 = r2_score(true_vol, pred_vol)
            mae = mean_absolute_error(true_vol, pred_vol)
            span = f"{te['timestamp'].iloc[0]:%Y-%m} -> {te['timestamp'].iloc[-1]:%Y-%m}"
            print(f"window {k+1} ({span}): R2 {r2:.3f} | MAE {mae:.5f} | n={len(true_vol)}")
            log.info("window %d R2 %.3f MAE %.5f", k + 1, r2, mae)

            all_true.append(true_vol); all_pred.append(pred_vol); all_ts.append(ts)
            model.save(OUT / "models" / f"brain_window{k+1}.keras")

        return np.concatenate(all_true), np.concatenate(all_pred), np.concatenate(all_ts)


# ---------------------------------------------------------------- Phase 5
class ResultsReport:
    def __init__(self, y_true, y_pred, ts):
        self.y_true, self.y_pred, self.ts = y_true, y_pred, ts

    def financial_metrics(self):
        from sklearn.metrics import r2_score, mean_absolute_error
        signal = (self.y_pred > np.median(self.y_pred)).astype(float)
        realized = self.y_true
        strat_ret = signal * realized - (1 - signal) * realized.mean()
        sharpe = strat_ret.mean() / (strat_ret.std() + 1e-12) * np.sqrt(252 * 26)
        cum = np.cumsum(strat_ret)
        mdd = (np.maximum.accumulate(cum) - cum).max()
        m = {
            "r2": float(r2_score(self.y_true, self.y_pred)),
            "mae": float(mean_absolute_error(self.y_true, self.y_pred)),
            "sharpe_vol_timing": float(sharpe),
            "max_drawdown": float(mdd),
            "n": int(len(self.y_true)),
        }
        with open(OUT / "reports" / "metrics.json", "w") as f:
            json.dump(m, f, indent=2)
        print("\n=== RESULTS ===")
        for k, v in m.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        return m

    def plot(self):
        idx = np.argsort(self.ts)
        step = max(1, len(idx) // 4000)
        idx = idx[::step]
        plt.figure(figsize=(13, 6))
        plt.plot(self.ts[idx], self.y_true[idx], lw=0.8, label="actual forward vol")
        plt.plot(self.ts[idx], self.y_pred[idx], lw=0.8, alpha=0.8, label="predicted")
        plt.title("QuantBrain — predicted vs actual forward realized vol (walk-forward)")
        plt.legend(); plt.tight_layout()
        p = OUT / "reports" / "pred_vs_actual.png"
        plt.savefig(p, dpi=150)
        print(f"saved {p}")


def main():
    print("=== QuantMind HGBNA — forward-volatility on vega stocks ===")
    engine = QuantDataEngine("_prepped_stk")
    df, feats = engine.build()
    print(f"rows {len(df)} | candidate features {len(feats)}")

    cycle = TrainingCycle(lookback=16, n_splits=4, top_n=15)
    y_true, y_pred, ts = cycle.run(df, feats)

    rep = ResultsReport(y_true, y_pred, ts)
    rep.financial_metrics()
    rep.plot()
    log.info("done")


if __name__ == "__main__":
    main()
