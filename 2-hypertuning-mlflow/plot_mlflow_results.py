import argparse
from pathlib import Path
import mlflow
import mlflow.tracking
import pandas as pd
import matplotlib.pyplot as plt


def load_runs_df(experiment_name: str):
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment {experiment_name} not found")

    runs = client.search_runs(exp.experiment_id)
    if not runs:
        raise ValueError(f"No runs found for experiment {experiment_name}")

    return pd.DataFrame([r.data.metrics | r.data.params for r in runs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", type=str, default="03_mlflow_with_norm_dropout_conv")
    ap.add_argument("--outdir", type=str, default="mlflow_charts")
    ap.add_argument("--tracking-dir", type=str,
                    default=str((Path(__file__).parent / "mlruns").resolve()),
                    help="Path to MLflow tracking dir (default=mlruns next to this script)")
    args = ap.parse_args()

    tracking_path = Path(args.tracking_dir).resolve()
    mlflow.set_tracking_uri(f"file://{tracking_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_runs_df(args.experiment)
    print(df.head())

    # Example: scatter plot learning rate vs val_accuracy
    if "lr" in df and "val_accuracy" in df:
        plt.scatter(df["lr"].astype(float), df["val_accuracy"].astype(float))
        plt.xscale("log")
        plt.xlabel("Learning rate")
        plt.ylabel("Validation accuracy")
        plt.title("LR vs Val Accuracy")
        plt.savefig(outdir / "lr_vs_val_acc.png")
        plt.close()


if __name__ == "__main__":
    main()