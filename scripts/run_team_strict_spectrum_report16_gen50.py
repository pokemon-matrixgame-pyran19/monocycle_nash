from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    targets = [t for t in TARGETS if args.only in (None, t.name)]

    for target in targets:
        started = time.perf_counter()
        code = run(MainConfigLoader(target.config_path))
        elapsed = time.perf_counter() - started
        print(f"[{target.name}] code={code} elapsed_sec={elapsed:.3f} config={target.config_path}")
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
