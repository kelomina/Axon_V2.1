import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def export_family_classifier(pkl_path: Path, out_path: Path) -> None:
    obj = pickle.loads(pkl_path.read_bytes())
    centroids = obj["centroids"]
    thresholds = obj["thresholds"]
    family_names = obj["family_names"]
    scaler = obj.get("scaler", None)

    cluster_ids = sorted(set(centroids.keys()) & set(thresholds.keys()) & set(family_names.keys()))
    centroids_list = [np.asarray(centroids[cid], dtype=np.float32).tolist() for cid in cluster_ids]

    thresholds_list = [float(thresholds[cid]) for cid in cluster_ids]
    family_names_list = [str(family_names[cid]) for cid in cluster_ids]

    scaler_mean = []
    scaler_scale = []
    if scaler is not None and hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        scaler_mean = np.asarray(scaler.mean_, dtype=np.float32).tolist()
        scaler_scale = np.asarray(scaler.scale_, dtype=np.float32).tolist()

    out = {
        "cluster_ids": [int(x) for x in cluster_ids],
        "centroids": centroids_list,
        "thresholds": thresholds_list,
        "family_names": family_names_list,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="hdbscan_cluster_results/family_classifier.pkl")
    parser.add_argument("--output", default="hdbscan_cluster_results/family_classifier.json")
    args = parser.parse_args()

    export_family_classifier(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
