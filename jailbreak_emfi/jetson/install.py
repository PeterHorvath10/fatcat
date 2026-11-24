"""Render the systemd units (and the sudoers rule) of one board from its config file.

    python3 -m jetson.install thor OUT_DIR      # uses config/thor.toml
    python3 -m jetson.install orin OUT_DIR      # uses config/orin.toml

Templates are jetson/<board>/systemd/*.in with @PLACEHOLDER@ fields; files without .in are copied. On
Thor one vllm unit is rendered per [models.*] section (named models.<key>.unit). Nothing is installed:
docs/jetson.md lists the commands to copy the files to /etc/systemd/system.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from common import config

JETSON = Path(__file__).resolve().parent
PLACEHOLDER = re.compile(r"@([A-Z0-9_]+)@")
CPU_PERFORMANCE_PRE = ('ExecStartPre=+/bin/sh -c "for p in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; '
                       'do echo performance > $$p; done"')
SYSTEMCTL = "/usr/bin/systemctl"


def render(text: str, values: dict[str, str], source: str) -> str:
    missing = sorted({m for m in PLACEHOLDER.findall(text) if m not in values})
    if missing:
        raise config.ConfigError(f"{source}: no value for {', '.join(missing)}")
    return PLACEHOLDER.sub(lambda m: values[m.group(1)], text)


def common_values(cfg: config.Config) -> dict[str, str]:
    return {"USER": cfg.get("system.user", str), "HOME": cfg.get("system.home", str),
            "PAPER_CODE": cfg.get("system.paper_code", str), "PYTHON": cfg.get("system.python", str)}


def thor_files(cfg: config.Config) -> dict[str, str]:
    """{output file name: content}."""
    values = common_values(cfg)
    models = list(cfg.section("models"))
    if not models:
        raise config.ConfigError("config/thor.toml has no [models.*] section")
    units = {key: cfg.get(f"models.{key}.unit", str).removesuffix(".service") for key in models}
    values["MODEL_UNITS"] = " ".join(f"{u}.service" for u in units.values())
    values["GPU_CLOCK_HZ"] = str(cfg.get("gpu_clock.gpc_hz", int))
    values["CPU_GOVERNOR_EXEC_START_PRE"] = (
        "# Fixed CPU clock (performance governor) so the reload timing is deterministic.\n" + CPU_PERFORMANCE_PRE
        if cfg.get("reload_trigger.cpu_performance_governor", bool) else "")
    values["SUDOERS_RULES"] = "\n".join(f"{values['USER']} ALL=(root) NOPASSWD: {SYSTEMCTL} restart {u}.service"
                                        for u in units.values())
    out: dict[str, str] = {}
    systemd_dir = JETSON / "thor" / "systemd"
    model_template = (systemd_dir / "vllm-model.service.in").read_text()
    for key, unit in units.items():
        others = [f"{u}.service" for k, u in units.items() if k != key]
        out[f"{unit}.service"] = render(model_template, {**values, "MODEL_KEY": key,
                                                         "CONFLICTS": " ".join(["gpu-memfault.service", *others])},
                                        "vllm-model.service.in")
    for path in sorted(systemd_dir.iterdir()):
        if path.name == "vllm-model.service.in":
            continue
        name = path.name.removesuffix(".in")
        out[name] = render(path.read_text(), values, path.name) if path.suffix == ".in" else path.read_text()
    return out


def orin_files(cfg: config.Config) -> dict[str, str]:
    values = {**common_values(cfg), "PA_MODULE": cfg.get("memfault.pa_module", str)}
    systemd_dir = JETSON / "orin" / "systemd"
    return {path.name.removesuffix(".in"): render(path.read_text(), values, path.name) if path.suffix == ".in"
            else path.read_text() for path in sorted(systemd_dir.iterdir())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("board", choices=("thor", "orin"))
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    cfg = config.load(f"{args.board}.toml")
    files = thor_files(cfg) if args.board == "thor" else orin_files(cfg)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (args.out / name).write_text(text)
        print(args.out / name)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except config.ConfigError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
