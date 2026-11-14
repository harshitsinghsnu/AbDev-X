"""Download the (gated) GDPa1 dataset from the HuggingFace Hub.

Usage:
    huggingface-cli login                      # one-time, after accepting dataset terms
    python -m abdev.data.download --config configs/default.yaml [--largest]

With --largest (default True via config), we list every CSV in the repo and fetch the one
with the most antibodies (largest file), so we always train on the fullest available release
rather than a competition subset.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def list_csvs(hf_repo: str):
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(repo_id=hf_repo, repo_type="dataset")
    return [f for f in files if f.lower().endswith(".csv")]


def _file_size(hf_repo: str, filename: str) -> int:
    from huggingface_hub import HfApi
    info = HfApi().get_paths_info(repo_id=hf_repo, paths=[filename], repo_type="dataset")
    return getattr(info[0], "size", 0) or 0


def pick_largest_csv(hf_repo: str) -> str:
    csvs = list_csvs(hf_repo)
    if not csvs:
        raise RuntimeError(f"No CSV files found in {hf_repo}")
    sizes = {f: _file_size(hf_repo, f) for f in csvs}
    best = max(sizes, key=sizes.get)
    print("Available CSVs (size bytes):")
    for f, s in sorted(sizes.items(), key=lambda kv: -kv[1]):
        print(f"  {'*' if f == best else ' '} {f}: {s:,}")
    return best


def download_file(hf_repo: str, filename: str, raw_dir: str) -> Path:
    from huggingface_hub import hf_hub_download
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    local = hf_hub_download(repo_id=hf_repo, filename=filename, repo_type="dataset",
                            local_dir=raw_dir)
    print(f"Downloaded {hf_repo}/{filename} -> {local}")
    return Path(local)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--file", default=None, help="explicit filename (overrides --largest)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    d = cfg["data"]

    if args.file:
        fname = args.file
    elif d.get("use_largest_csv", True):
        fname = pick_largest_csv(d["hf_repo"])
    else:
        fname = d["csv_filename"]

    path = download_file(d["hf_repo"], fname, d["raw_dir"])
    # Record which file we actually used so the loader can find it.
    (Path(d["raw_dir"]) / "ACTIVE_CSV.txt").write_text(path.name, encoding="utf-8")
    print(f"Active CSV recorded: {path.name}")


if __name__ == "__main__":
    main()
