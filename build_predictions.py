"""Step 7, part 1: write six predictions files for the official harness.

Patch sets: a1, a2, a3 (from logs/first), S, T, ST (submitted patches).
Each file is predictions/<set>.jsonl with one line per instance:
    {"instance_id": ..., "model_name_or_path": <set>, "model_patch": ...}

The model_name_or_path field is set to the set name, so that the harness
writes each set's reports into its own folder.
"""

import argparse
import json

from common import PRED_DIR, load_log, load_sampled_instances

SETS = ["a1", "a2", "a3", "S", "T", "ST"]


def patch_for(set_name: str, iid: str):
    if set_name in ("a1", "a2", "a3"):
        log = load_log("first", iid)
        if not log or not log.get("attempts"):
            return None
        return log["attempts"][int(set_name[1]) - 1]["patch"]
    log = load_log(set_name, iid)
    if not log or "submitted_patch" not in log:
        return None
    return log["submitted_patch"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sets", nargs="*", default=SETS)
    args = ap.parse_args()

    PRED_DIR.mkdir(exist_ok=True)
    ids = [i["instance_id"] for i in load_sampled_instances(args.limit)]
    for s in args.sets:
        rows, missing = [], []
        for iid in ids:
            p = patch_for(s, iid)
            if p is None:
                missing.append(iid)
                continue
            rows.append({"instance_id": iid, "model_name_or_path": s, "model_patch": p})
        out = PRED_DIR / f"{s}.jsonl"
        out.write_text("".join(json.dumps(r) + "\n" for r in rows))
        empty = sum(1 for r in rows if not r["model_patch"].strip())
        print(f"{s}: {len(rows)} predictions ({empty} empty patches) -> {out}"
              + (f"; MISSING logs for {len(missing)}: {missing[:5]}" if missing else ""))


if __name__ == "__main__":
    main()
