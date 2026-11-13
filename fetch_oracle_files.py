"""Fetch the oracle file for every sampled instance.

The oracle file is the one file the reference fix changed. Its path comes from
the gold patch header; its content is read from the instance's Docker image,
where /testbed is checked out at base_commit. This gives the exact text the
hidden tests will patch against. The gold patch content is never used.

Writes data/oracle_files/<instance_id>.txt and data/oracle_paths.json.
Run with --limit 50 for the pilot. Already fetched files are skipped.
"""

import argparse
import json

from common import DATA_DIR, ORACLE_DIR, Container, load_sampled_instances, oracle_file_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="first N sampled instances")
    args = ap.parse_args()

    ORACLE_DIR.mkdir(parents=True, exist_ok=True)
    paths_file = DATA_DIR / "oracle_paths.json"
    paths = json.loads(paths_file.read_text()) if paths_file.exists() else {}

    for inst in load_sampled_instances(args.limit):
        iid = inst["instance_id"]
        out = ORACLE_DIR / f"{iid}.txt"
        path = oracle_file_path(inst)
        paths[iid] = path
        if out.exists():
            continue
        print(f"{iid}: {path}")
        with Container(inst, name_suffix="oracle") as c:
            content = c.read_file(f"/testbed/{path}")
        out.write_text(content)
        paths_file.write_text(json.dumps(paths, indent=2))
    paths_file.write_text(json.dumps(paths, indent=2))
    print(f"done; {len(paths)} paths in {paths_file}")


if __name__ == "__main__":
    main()
