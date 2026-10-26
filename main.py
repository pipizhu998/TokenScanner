import argparse
import json
from pathlib import Path

import yaml


def run_scan(root, options):
    from scan import scan_model

    name = options.get("name") or options["adapter"].name
    return scan_model(output_dir=root / "scans" / name, cache_dir=root, **options)


def run_thresholding(root, options):
    from thresholding import fit_thresholds, read_model_scores, save_thresholds

    thresholds = fit_thresholds(read_model_scores(options["input"]), options.get("quantile", 0.9))
    output = root / "thresholds/thresholds.json"
    save_thresholds(thresholds, output)
    return {"thresholds": str(output), "groups": len(thresholds["groups"])}


def run_eval(root, options):
    from eval import evaluate

    options["thresholds"] = options.get("thresholds") or root / "thresholds/thresholds.json"
    return evaluate(output_dir=root / "eval", **options)


def run(config_path: Path | str) -> dict:
    config_path = Path(config_path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    def resolve(value):
        return (config_path.parent / Path(value).expanduser()).resolve()

    task = config["task"]
    root = resolve(config.get("output_dir", "output"))
    options = config[task].copy()
    for key in ("adapter", "base_model", "input", "thresholds"):
        if options.get(key) is not None:
            options[key] = resolve(options[key])

    tasks = {"scan": run_scan, "thresholding": run_thresholding, "eval": run_eval}
    return tasks[task](root, options)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TokenScanner from a YAML configuration.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2))


if __name__ == "__main__":
    main()
