from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from monocycle_nash.analysis.app.experiment_team_strict_spectrum import run
from monocycle_nash.runtime.infra.loader.main_config import MainConfigLoader


@dataclass(frozen=True)
class RunTarget:
    name: str
    config_path: str


TARGETS: tuple[RunTarget, ...] = (
    RunTarget("baseline", "data/run_config/experiment_team_strict_spectrum_report16_50_baseline.toml"),
    RunTarget("power_wide", "data/run_config/experiment_team_strict_spectrum_report16_50_power_wide.toml"),
    RunTarget("vector_wide", "data/run_config/experiment_team_strict_spectrum_report16_50_vector_wide.toml"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run character_count=16, generation_count=50 strict-spectrum experiments.",
    )
    parser.add_argument(
        "--only",
        choices=[t.name for t in TARGETS],
        help="Run only one condition.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory where run outputs are written (default: results).",
    )
    return parser.parse_args()


def _list_run_ids(results_dir: Path) -> set[int]:
    if not results_dir.exists():
        return set()
    run_ids: set[int] = set()
    for child in results_dir.iterdir():
        if child.is_dir() and child.name.isdigit():
            run_ids.add(int(child.name))
    return run_ids


def _load_experiment_summary(results_dir: Path, run_id: int) -> dict:
    path = results_dir / str(run_id) / "output" / "team_strict_spectrum_experiment.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["summary"]


def main() -> int:
    args = parse_args()
    targets = [t for t in TARGETS if args.only in (None, t.name)]
    results_dir = Path(args.results_dir)
    records: list[dict] = []

    for target in targets:
        before = _list_run_ids(results_dir)
        started = time.perf_counter()
        code = run(MainConfigLoader(target.config_path))
        elapsed = time.perf_counter() - started
        after = _list_run_ids(results_dir)
        created = sorted(after - before)
        run_id = created[-1] if created else None
        summary = _load_experiment_summary(results_dir, run_id) if run_id is not None else None

        print(
            f"[{target.name}] code={code} elapsed_sec={elapsed:.3f} "
            f"run_id={run_id} config={target.config_path}"
        )
        records.append(
            {
                "target": target.name,
                "config_path": target.config_path,
                "code": code,
                "elapsed_sec": elapsed,
                "run_id": run_id,
                "summary": summary,
            }
        )
        if code != 0:
            return code

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    aggregate = {
        "script": "run_team_strict_spectrum_report16_gen50.py",
        "created_at_utc": timestamp,
        "records": records,
    }
    out_latest = results_dir / "team_strict_spectrum_report16_gen50_latest.json"
    out_timestamped = results_dir / f"team_strict_spectrum_report16_gen50_{timestamp}.json"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_latest.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    out_timestamped.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"aggregate_written={out_latest}")
    print(f"aggregate_written={out_timestamped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
