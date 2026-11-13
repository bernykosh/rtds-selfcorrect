"""Check why patches fail to apply, without Docker.

For every attempt in logs/first, the script builds a temporary git repository
that holds only the oracle file at its path. Then:
  1. git apply --check on the patch as the model wrote it
  2. git apply --check on the patch after repair_hunk_headers()

The script prints the git error messages grouped by type, and how many
patches the header repair would rescue. It changes no logs.
"""

import collections
import json
import subprocess
import tempfile
from pathlib import Path

from common import LOG_DIR, ORACLE_DIR, repair_hunk_headers


def git_check(oracle_path: str, oracle_text: str, patch: str) -> str:
    """Return '' when the patch applies, else the first git error line."""
    if not patch.strip():
        return "empty patch"
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        f = repo / oracle_path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(oracle_text)
        (repo / "p.diff").write_text(patch)
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        subprocess.run(["git", "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "base"], cwd=repo, check=True)
        r = subprocess.run(["git", "apply", "--check", "p.diff"], cwd=repo, capture_output=True, text=True)
        if r.returncode == 0:
            return ""
        lines = [l for l in r.stderr.strip().split("\n") if l.startswith("error:")]
        return lines[0] if lines else r.stderr.strip().split("\n")[0]


def classify(err: str) -> str:
    if not err:
        return "applies"
    if "corrupt patch" in err:
        return "corrupt patch (bad hunk header or body)"
    if "does not apply" in err or "patch failed" in err:
        return "content mismatch (context lines differ from file)"
    if "No such file" in err or "does not exist" in err:
        return "wrong path"
    return err[:60]


def main():
    before = collections.Counter()
    after = collections.Counter()
    examples = {}
    for f in sorted((LOG_DIR / "first").glob("*.json")):
        log = json.loads(f.read_text())
        if not log.get("attempts"):
            continue
        oracle = (ORACLE_DIR / f"{log['instance_id']}.txt").read_text()
        for a in log["attempts"]:
            e1 = git_check(log["oracle_file"], oracle, a["patch"])
            c1 = classify(e1)
            before[c1] += 1
            examples.setdefault(c1, (log["instance_id"], a["attempt"], e1))
            e2 = git_check(log["oracle_file"], oracle, repair_hunk_headers(a["patch"]))
            after[classify(e2)] += 1
    print("As written by the model:")
    for k, v in before.most_common():
        print(f"  {v:4d}  {k}")
    print("After repair_hunk_headers():")
    for k, v in after.most_common():
        print(f"  {v:4d}  {k}")
    print("One example per class (instance, attempt, git message):")
    for k, v in examples.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
