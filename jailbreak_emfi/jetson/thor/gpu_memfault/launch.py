"""Start gpu_memfault with the [memfault] settings of config/thor.toml (ExecStart of gpu-memfault.service).

    sudo python3 -m jetson.thor.gpu_memfault.launch [--free-run GAP_S]
"""

from __future__ import annotations

import argparse
import os

from common import config


def memfault_command(cfg: config.Config, free_run_s: float | None = None) -> list[str]:
    command = [cfg.get("memfault.binary"),
               "--gib", str(cfg.get("memfault.gib", float)),
               "--window", str(cfg.get("memfault.window_s", float)),
               "--pattern", cfg.get("memfault.pattern"),
               "--mem", cfg.get("memfault.mem"),
               "--log", cfg.get("memfault.log")]
    if free_run_s is None:
        command += ["--host", cfg.get("network.listen_host"), "--port", str(cfg.get("network.listen_port", int))]
    else:
        command += ["--free-run", str(free_run_s)]
    if cfg.get("memfault.ecc_polling", bool):
        command += ["--bar0", cfg.get("memfault.bar0")]
    else:
        command.append("--no-ecc")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--free-run", type=float, metavar="GAP_S",
                        help="no controller: one verify window every GAP_S idle seconds (scope setup)")
    args = parser.parse_args()
    command = memfault_command(config.load("thor.toml"), args.free_run)
    print("exec:", " ".join(command), flush=True)
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
