"""Bundle the exported champion into the model.tar.gz layout SageMaker expects (files at the archive root)."""
from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


MODEL_FILE, META_FILE, REFERENCE_FILE = "model.joblib", "model_meta.json", "reference_sample.csv"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="models")
    p.add_argument("--out", default="build/model.tar.gz")
    a = p.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        for name in (MODEL_FILE, META_FILE, REFERENCE_FILE):
            src = Path(a.model_dir) / name
            if not src.exists():
                raise SystemExit(f"missing {src} - run `make train` first")
            tar.add(src, arcname=name)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
