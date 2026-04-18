from datetime import datetime

import pytest

from monocycle_nash.domain.experiment_run import ExperimentRun, VersionPort


class TestExperimentRun:
    def test_can_create_with_minimum_metadata(self):
        created_at = datetime(2026, 4, 18, 5, 0, 0)
        run = ExperimentRun(
            serial_id=1,
            elapsed_seconds=12.5,
            created_at=created_at,
            version="abc123",
        )

        assert run.serial_id == 1
        assert run.elapsed_seconds == pytest.approx(12.5)
        assert run.created_at == created_at
        assert run.version == "abc123"

    def test_rejects_non_positive_serial_id(self):
        with pytest.raises(ValueError, match="serial_id"):
            ExperimentRun(
                serial_id=0,
                elapsed_seconds=1.0,
                created_at=datetime.now(),
                version="abc123",
            )

    def test_rejects_negative_elapsed_seconds(self):
        with pytest.raises(ValueError, match="elapsed_seconds"):
            ExperimentRun(
                serial_id=1,
                elapsed_seconds=-0.1,
                created_at=datetime.now(),
                version="abc123",
            )

    def test_rejects_empty_version(self):
        with pytest.raises(ValueError, match="version"):
            ExperimentRun(
                serial_id=1,
                elapsed_seconds=1.0,
                created_at=datetime.now(),
                version=" ",
            )


def test_version_port_requires_implementation():
    with pytest.raises(TypeError):
        VersionPort()
