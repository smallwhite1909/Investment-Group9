"""Train the fixed Group 9 models and reproduce their portfolios from external inputs."""

import argparse
import json
from pathlib import Path

from backtest import add_lasso_backtests, run_backtests
from data import load_data
from models import train_lasso, train_models
from report import build_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "lasso", "report"], default="all",
                        help="Train all models, add only Lasso to existing outputs, or rebuild documents")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing the prepared market arrays")
    parser.add_argument("--features", type=Path, required=True,
                        help="Prepared float32 date x stock x 16 factor panel")
    parser.add_argument("--actions", type=Path, required=True,
                        help="External corporate-action CSV; see data_dictionary.md")
    parser.add_argument("--output", type=Path,
                        default=Path("/data/share/XingyuLargeFiles/Investment-Group9"),
                        help="External output directory, outside the code repository")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[1] / "config.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output.resolve()
    if output == Path(__file__).resolve().parents[1] or Path(__file__).resolve().parents[1] in output.parents:
        parser.error("--output must be outside this repository")
    output.mkdir(parents=True, exist_ok=True)
    data = load_data(args.data_dir, args.features, args.actions, config)
    print(f"Loaded {len(data['days'])} sessions and {len(data['codes'])} instrument slots", flush=True)
    if args.stage == "all":
        result = train_models(data, config, output / "models")
        run_backtests(data, result, output, config)
    elif args.stage == "lasso":
        result = train_lasso(data, config, output / "models")
        add_lasso_backtests(data, result, output, config)
    build_report(data, config, output, Path(__file__).resolve().parents[1])
    print(f"Finished: {output}", flush=True)


if __name__ == "__main__":
    main()
