"""Print the glitch parameters of the trial that caused the hit: the last trial-log row without an error
(the CHANGED check stops the worker, so the row after it fails with a closed socket). With the watcher's
HIT line, a hits CSV and the target tensor, also append the hit (parameters + changed tensors) to the CSV.

Usage: hit_delay.py TRIALS_CSV [HIT_LINE HITS_CSV TARGET]"""

import csv
import json
import os
import re
import sys
import time

from host.emfi.glitcher import counts_to_ms

HITS_FIELDS = ["time", "round", "trial", "primary_ms", "voltage", "duration",
               "on_target", "tensors", "changed_elems", "lost_write_elems", "garbage_elems"]


def main():
    rows = [r for r in csv.DictReader(open(sys.argv[1])) if not r["error"]]
    r = rows[-1]
    primary_ms = counts_to_ms(int(r["primary_delay"]))
    print(f"trial {r['trial_index']}: primary {primary_ms:g} ms after the edge, "
          f"{r['glitch_voltage']} V, duration {r['primary_duration']}")
    if len(sys.argv) <= 4:
        return
    hit, path, target = sys.argv[2], sys.argv[3], sys.argv[4]
    report = json.loads(hit.split("changed: ", 1)[1])
    names = {t["name"] for t in report["tensors"]}
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(HITS_FIELDS)
        w.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), re.search(r"trials(\d+)\.csv$", sys.argv[1]).group(1),
                    r["trial_index"], f"{primary_ms:g}", r["glitch_voltage"], r["primary_duration"],
                    target + ".weight" in names and names <= {target + ".weight", target + ".weight_scale"},
                    ";".join(t["name"].removeprefix("model.") for t in report["tensors"]),
                    ";".join(str(t["changed_elems"]) for t in report["tensors"]),
                    report.get("lost_write_elems", ""), report.get("garbage_elems", "")])


if __name__ == "__main__":
    main()
