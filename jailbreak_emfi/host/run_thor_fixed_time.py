"""Thor: vLLM JailbreakBench at a fixed delay (prompt slots), probe at one position.

    PYTHONPATH=<jailbreak_emfi> python host/run_thor_fixed_time.py [--experiment experiments/thor_fixed_time.toml] [options]

See docs/host.md; settings in config/host.toml and config/experiments/thor_fixed_time.toml."""
from host.emfi import cli

if __name__ == "__main__":
    raise SystemExit(cli.main(__doc__.splitlines()[0], "experiments/thor_fixed_time.toml"))
