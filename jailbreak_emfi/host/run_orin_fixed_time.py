"""Orin: Ollama JailbreakBench at a fixed delay window, probe at one position.

    PYTHONPATH=<jailbreak_emfi> python host/run_orin_fixed_time.py [--experiment experiments/orin_fixed_time.toml] [options]

See docs/host.md; settings in config/host.toml and config/experiments/orin_fixed_time.toml."""
from host.emfi import cli

if __name__ == "__main__":
    raise SystemExit(cli.main(__doc__.splitlines()[0], "experiments/orin_fixed_time.toml"))
