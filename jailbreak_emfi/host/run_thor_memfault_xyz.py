"""Thor: XYZ scan with the gpu-memfault memory-check workload.

    PYTHONPATH=<jailbreak_emfi> python host/run_thor_memfault_xyz.py [--experiment experiments/thor_memfault_xyz.toml] [options]

See docs/host.md; settings in config/host.toml and config/experiments/thor_memfault_xyz.toml."""
from host.emfi import cli

if __name__ == "__main__":
    raise SystemExit(cli.main(__doc__.splitlines()[0], "experiments/thor_memfault_xyz.toml"))
