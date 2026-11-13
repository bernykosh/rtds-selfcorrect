"""Step 2 of the protocol: sample 100 instances from SWE-bench Lite.

Rule (protocol sheet):
- seed 2026
- stratified by repository in proportion to repository counts, rounded,
  remainder assigned to the largest repository
- order shuffled with the same seed
- pilot = first 50 in sampled order

Writes data/instances.json with the instance IDs in sampled order, the
repository counts in the full set and in the sample, and the seed.
"""

import json
import random
from collections import Counter

from common import DATA_DIR, INSTANCES_FILE, SAMPLE_SIZE, SEED, load_lite_dataset


def main():
    rows = load_lite_dataset()
    print(f"loaded {len(rows)} instances from the test split")
    ids_by_repo = {}
    for r in rows:
        ids_by_repo.setdefault(r["repo"], []).append(r["instance_id"])
    for repo in ids_by_repo:
        ids_by_repo[repo].sort()  # deterministic order before shuffling

    total = len(rows)
    global SAMPLE_SIZE
    if total < SAMPLE_SIZE:
        print(f"WARNING: only {total} instances available; sampling all of them")
        SAMPLE_SIZE = total
    full_counts = {repo: len(ids) for repo, ids in ids_by_repo.items()}

    # Proportional allocation, rounded to the nearest whole number.
    alloc = {repo: round(SAMPLE_SIZE * n / total) for repo, n in full_counts.items()}
    largest = max(full_counts, key=full_counts.get)
    alloc[largest] += SAMPLE_SIZE - sum(alloc.values())  # remainder to the largest repo
    assert sum(alloc.values()) == SAMPLE_SIZE, alloc

    rng = random.Random(SEED)
    sampled = []
    for repo in sorted(ids_by_repo):
        ids = ids_by_repo[repo]
        k = alloc[repo]
        sampled.extend(rng.sample(ids, k))
    rng.shuffle(sampled)

    DATA_DIR.mkdir(exist_ok=True)
    INSTANCES_FILE.write_text(json.dumps({
        "seed": SEED,
        "sample_size": SAMPLE_SIZE,
        "repo_counts_full": dict(sorted(full_counts.items())),
        "repo_counts_sample": dict(sorted(Counter(r.split("__")[0] for r in sampled).items())),
        "allocation": dict(sorted(alloc.items())),
        "instance_ids": sampled,
    }, indent=2))
    print(f"wrote {INSTANCES_FILE} with {len(sampled)} instance IDs")
    for repo, k in sorted(alloc.items()):
        print(f"  {repo:40s} {full_counts[repo]:4d} in full set -> {k:3d} sampled")


if __name__ == "__main__":
    main()
