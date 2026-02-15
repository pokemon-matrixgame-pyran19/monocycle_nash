from __future__ import annotations

import json
import traceback

import numpy as np

from monocycle_nash.character.domain import Character, MatchupVector
from monocycle_nash.matrix.builder import PayoffMatrixBuilder
from monocycle_nash.runmeta.artifact_store import RunArtifactStore
from monocycle_nash.runmeta.db import SQLiteConnectionFactory, migrate
from monocycle_nash.runmeta.repositories import ProjectsRepository, RunsRepository
from monocycle_nash.runmeta.service import RunSessionService
from monocycle_nash.solver.selector import SolverSelector


def run_domain_calculation() -> dict[str, object]:
    characters = [
        Character(0.50, MatchupVector(0.4, 0.0), "c1"),
        Character(0.50, MatchupVector(0.0, 0.6), "c2"),
        Character(0.50, MatchupVector(-0.2, -0.2), "c3"),
    ]

    matrix = PayoffMatrixBuilder.from_characters(characters, labels=["c1", "c2", "c3"])
    equilibrium = SolverSelector().solve(matrix)
    return {
        "matrix": np.asarray(matrix.matrix).tolist(),
        "strategy_ids": equilibrium.strategy_ids,
        "probabilities": equilibrium.probabilities.tolist(),
    }


def main() -> int:
    conn = SQLiteConnectionFactory("runmeta.sqlite3").connect()
    migrate(conn)

    projects = ProjectsRepository(conn)
    runs = RunsRepository(conn)
    artifacts = RunArtifactStore("result")
    session = RunSessionService(run_repository=runs, artifact_store=artifacts)

    project_id = projects.add("monocycle_nash")
    run_id = session.start(project_id)

    try:
        output = run_domain_calculation()
        artifacts.write_output_file(run_id, "result.json", json.dumps(output, ensure_ascii=False, indent=2))
        session.finish_success(run_id, {"output_file": "output/result.json"})
        return 0
    except KeyboardInterrupt:
        artifacts.write_log_file(run_id, "killed.log", "Run interrupted by user")
        session.finish_killed(run_id, {"reason": "keyboard_interrupt"})
        return 130
    except Exception as exc:  # noqa: BLE001
        artifacts.write_log_file(run_id, "error.log", traceback.format_exc())
        session.finish_fail(run_id, {"error": str(exc), "log_file": "logs/error.log"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
