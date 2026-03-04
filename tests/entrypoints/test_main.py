from __future__ import annotations

import pytest

import monocycle_nash.main as main_mod


class _FakeConfigLoader:
    def __init__(self, features: list[str]) -> None:
        self._features = features

    def load_features(self) -> list[str]:
        return self._features


def test_main_runs_compare_payoff_and_returns_non_zero_code(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_loader = _FakeConfigLoader(["compare_payoff"])
    monkeypatch.setattr(main_mod, "MainConfigLoader", lambda: fake_loader)

    called: list[_FakeConfigLoader] = []

    def _fake_compare_run(config_loader: _FakeConfigLoader) -> int:
        called.append(config_loader)
        return 7

    monkeypatch.setattr(main_mod, "run_compare_payoff", _fake_compare_run)

    assert main_mod.main() == 7
    assert called == [fake_loader]


def test_main_raises_value_error_with_unsupported_feature_name(monkeypatch: pytest.MonkeyPatch) -> None:
    unsupported_feature = "compare_payoff_next"
    monkeypatch.setattr(main_mod, "MainConfigLoader", lambda: _FakeConfigLoader([unsupported_feature]))

    with pytest.raises(ValueError, match=unsupported_feature):
        main_mod.main()


def test_main_runs_compare_random_approximation(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_loader = _FakeConfigLoader(["compare_random_approximation"])
    monkeypatch.setattr(main_mod, "MainConfigLoader", lambda: fake_loader)

    called: list[_FakeConfigLoader] = []

    def _fake_run(config_loader: _FakeConfigLoader) -> int:
        called.append(config_loader)
        return 0

    monkeypatch.setattr(main_mod, "run_compare_random_approximation", _fake_run)

    assert main_mod.main() == 0
    assert called == [fake_loader]
