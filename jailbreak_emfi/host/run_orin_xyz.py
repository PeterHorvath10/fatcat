"""Orin: XYZ scan over the chip, per-prompt down_proj windows (memory check / Ollama JailbreakBench).

    PYTHONPATH=<jailbreak_emfi> python host/run_orin_xyz.py [--experiment experiments/orin_xyz.toml] [options]

See docs/host.md; settings in config/host.toml and config/experiments/orin_xyz.toml."""
from host.emfi import cli

if __name__ == "__main__":
    raise SystemExit(cli.main(__doc__.splitlines()[0], "experiments/orin_xyz.toml"))
