"""Training pipeline: 3 candidates -> MLflow tracking -> quality gate -> registry (champion alias) -> export.

    python -m src.train                # full run (RandomizedSearchCV 15x5)
    python -m src.train --quick        # CI smoke run
    MLFLOW_TRACKING_URI=http://localhost:5000 python -m src.train
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from sklearn.model_selection import RandomizedSearchCV, train_test_split  # noqa: E402

from src.artifacts import build_meta, save_artifacts  # noqa: E402
from src.config import (  # noqa: E402
    CHAMPION_ALIAS, DATA_PATH, EXPERIMENT_NAME, EXPORT_DIR, META_FILE, REFERENCE_FILE,
    REGISTERED_MODEL_NAME, ROOT, SEED, TARGET, TEST_SIZE,
)
from src.features import clean, fit_top_countries, prepare_features  # noqa: E402
from src.pipeline import get_candidates, regression_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(DATA_PATH))
    p.add_argument("--export-dir", default=str(EXPORT_DIR))
    p.add_argument("--min-r2", type=float, default=0.5, help="quality gate: fail if best test R2 is below this")
    p.add_argument("--quick", action="store_true", help="tiny hyper-parameter search (for CI)")
    return p.parse_args()


def _pred_vs_actual(y_true, y_pred, title: str):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(y_true, y_pred, s=8, alpha=0.5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1)
    ax.set(xlabel="actual", ylabel="predicted", title=title)
    fig.tight_layout()
    return fig


def _register_and_promote(client: MlflowClient, best: dict) -> tuple[str, bool]:
    """Register the best run; promote to `champion` only if it beats the current champion's RMSE."""
    mv = mlflow.register_model(f"runs:/{best['run_id']}/model", REGISTERED_MODEL_NAME)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, mv.version, "test_rmse", str(best["metrics"]["rmse"]))
    champion_rmse = None
    try:
        champ = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
        if champ.version != mv.version:
            champion_rmse = float(champ.tags.get("test_rmse", "inf"))
    except MlflowException:
        pass  # no champion yet
    promoted = champion_rmse is None or best["metrics"]["rmse"] <= champion_rmse
    if promoted:
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, mv.version)
    return mv.version, promoted


def _export_champion(client: MlflowClient, export_dir: Path) -> dict:
    """Pull whatever currently holds the `champion` alias out of the registry into deployable files."""
    champ = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    pipeline = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}")
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(client.download_artifacts(champ.run_id, "serving", tmp))
        meta = json.loads((local / META_FILE).read_text())
        meta["registry"] = {"name": REGISTERED_MODEL_NAME, "version": champ.version, "run_id": champ.run_id}
        save_artifacts(export_dir, pipeline, meta, pd.read_csv(local / REFERENCE_FILE))
    return meta


def main() -> int:
    args = parse_args()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"))
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    df = clean(pd.read_csv(args.data))
    top_countries = fit_top_countries(df)
    X, y = prepare_features(df, top_countries), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    data_md5 = hashlib.md5(Path(args.data).read_bytes()).hexdigest()

    results: list[dict] = []
    with mlflow.start_run(run_name=f"training-{datetime.now():%Y%m%d-%H%M%S}"):
        mlflow.log_params({"n_rows": len(df), "test_size": TEST_SIZE, "seed": SEED, "data_md5": data_md5})
        for name, estimator in get_candidates(quick=args.quick).items():
            with mlflow.start_run(run_name=name, nested=True) as run:
                estimator.fit(X_train, y_train)
                is_search = isinstance(estimator, RandomizedSearchCV)
                model = estimator.best_estimator_ if is_search else estimator
                if is_search:
                    mlflow.log_params({k.replace("regressor__", "rf_"): v for k, v in estimator.best_params_.items()})
                    mlflow.log_metric("cv_r2", float(estimator.best_score_))
                test_preds, train_preds = model.predict(X_test), model.predict(X_train)
                metrics = regression_metrics(y_test, test_preds)
                train_metrics = regression_metrics(y_train, train_preds)
                mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
                mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
                mlflow.log_metric("r2_gap", train_metrics["r2"] - metrics["r2"])
                mlflow.set_tag("algorithm", name)
                mlflow.log_figure(_pred_vs_actual(y_test, test_preds, name), "plots/pred_vs_actual.png")
                plt.close("all")
                mlflow.sklearn.log_model(model, "model", signature=infer_signature(X_test, test_preds),
                                         input_example=X_test.head(3),skops_trusted_types=["numpy.dtype","sklearn.tree._tree.Tree"],)
                results.append({"name": name, "run_id": run.info.run_id, "metrics": metrics,
                                "test_preds": test_preds})
                print(f"[{name}] R2={metrics['r2']:.4f} MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f}")

        best = min(results, key=lambda r: r["metrics"]["rmse"])
        mlflow.set_tags({"best_model": best["name"], "best_run_id": best["run_id"]})
        mlflow.log_metric("best_test_r2", best["metrics"]["r2"])
        mlflow.log_metric("best_test_rmse", best["metrics"]["rmse"])

        # ---- quality gate ------------------------------------------------------------------
        if best["metrics"]["r2"] < args.min_r2:
            mlflow.set_tag("quality_gate", "failed")
            print(f"QUALITY GATE FAILED: best R2 {best['metrics']['r2']:.3f} < {args.min_r2}", file=sys.stderr)
            return 1
        mlflow.set_tag("quality_gate", "passed")

        # ---- serving bundle (meta + drift reference) is stored on the winning run ------------
        meta, reference_sample = build_meta(best["name"], best["metrics"],
                                            [{"name": r["name"], **r["metrics"]} for r in results],
                                            top_countries, X_test, best["test_preds"])
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / META_FILE).write_text(json.dumps(meta))
            reference_sample.to_csv(Path(tmp) / REFERENCE_FILE, index=False)
            client.log_artifacts(best["run_id"], tmp, artifact_path="serving")

        version, promoted = _register_and_promote(client, best)
        print(f"Registered {REGISTERED_MODEL_NAME} v{version} ({'PROMOTED to champion' if promoted else 'kept as challenger'})")

    exported = _export_champion(client, Path(args.export_dir))
    print(f"Exported champion v{exported['registry']['version']} ({exported['algorithm']}) -> {args.export_dir}")
    # keep a copy of the summary for CI artifacts / README badges
    Path(args.export_dir, "training_summary.json").write_text(json.dumps(
        {"champion": exported["algorithm"], "metrics": exported["metrics"], "candidates": exported["candidates"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
