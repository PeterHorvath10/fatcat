"""Exit 0 if the HIT line's weight report is a full wipe of TARGET: more than MIN_ELEMS elements of
TARGET.weight changed and nothing but TARGET.weight(_scale) changed (a few scattered 16-byte blocks are not
a wipe). Also prints the changed tensor names.

Usage: is_full_wipe.py HIT_LINE TARGET MIN_ELEMS"""

import json
import sys

if __name__ == "__main__":
    hit, target, min_elems = sys.argv[1], sys.argv[2], int(sys.argv[3])
    d = {t["name"]: t["changed_elems"] for t in json.loads(hit.split("changed: ", 1)[1])["tensors"]}
    print(",".join(d))
    full = d.get(target + ".weight", 0) > min_elems and set(d) <= {target + ".weight", target + ".weight_scale"}
    sys.exit(0 if full else 1)
