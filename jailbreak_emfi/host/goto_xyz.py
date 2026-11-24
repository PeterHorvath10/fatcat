"""Move the EMFI XYZ stage to one chip position (stage coordinates, as in the scan logs' "Position:" lines).

    PYTHONPATH=<jailbreak_emfi> python host/goto_xyz.py X Y [Z]       # stage name: stage.table_name in host.toml
"""
import sys

from common import config
from host.emfi import stage

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    pos = [float(v) for v in sys.argv[1:4]]
    pos += [0.0] * (3 - len(pos))
    stage.goto(config.load("host.toml").get("stage.table_name"), pos)
