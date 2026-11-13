"""Step 8 and 9: compute every metric, test and interval; export coding.csv.

Inputs:
- logs/first, logs/S, logs/T, logs/ST   (model runs)
- logs/run_evaluation/<RUN_ID>/<set>/<instance_id>/report.json  (harness)

Outputs:
- results.json     all numbers, with the instance-level table they come from
- results.md       the same numbers as readable tables
- coding.csv       one row per instance where the submitted patch differs from a1,
                   with the fields the qualitative coder fills in

Statistics (protocol sheet):
- exact McNemar test on the discordant pairs, four planned comparisons
  S vs B, T vs S, T vs B, ST vs T; Holm correction; alpha 0.05
- 95 percent percentile bootstrap over instances, 10,000 resamples, seed 2026,
  on resolve-rate differences and on flip shares
"""

import argparse
import csv
import json
import random
from math import comb
from pathlib import Path

from common import LOG_DIR, ROOT, load_log, patches_equal

RUN_ID = "study2026"
COMPARISONS = [("S", "B", "SQ1"), ("T", "S", "SQ2"), ("T", "B", "SQ2"), ("ST", "T", "SQ3")]
SETS = ["a1", "a2", "a3", "S", "T", "ST"]
BOOT_N = 10_000
SEED = 2026
ALPHA = 0.05


# ---------------------------------------------------------------------------
# Harness results
# ---------------------------------------------------------------------------


def harness_report(run_id: str, set_name: str, iid: str):
    d = LOG_DIR / "run_evaluation" / run_id / set_name / iid
    p = d / "report.json"
    if not p.exists():
        if (d / "run_instance.log").exists():
            # The harness ran the instance but wrote no report. This happens when
            # the patch could not be applied: the harness raises an error and
            # stops. The SWE-bench score for such an instance is unresolved.
            log = (d / "run_instance.log").read_text(errors="replace")
            apply_failed = "Patch Apply Failed" in log or "patch failed" in log.lower()
            return {"resolved": False, "applied": False, "infra_failure": not apply_failed,
                    "harness_error": "patch failed to apply" if apply_failed else "other error; see run_instance.log"}
        if patch_is_empty(set_name, iid):
            return {"resolved": False, "applied": False, "infra_failure": False, "harness_error": "empty patch"}
        return None
    data = json.loads(p.read_text())
    r = data.get(iid, data)
    return {"resolved": bool(r.get("resolved")), "applied": bool(r.get("patch_successfully_applied")),
            "infra_failure": bool(r.get("infra_failure"))}


_pred_cache = {}


def patch_is_empty(set_name: str, iid: str) -> bool:
    """The harness writes no report for an empty patch; the protocol keeps it
    in the data as unresolved."""
    if set_name not in _pred_cache:
        f = ROOT / "predictions" / f"{set_name}.jsonl"
        _pred_cache[set_name] = {json.loads(l)["instance_id"]: json.loads(l)["model_patch"] for l in f.read_text().splitlines()} if f.exists() else {}
    return not _pred_cache[set_name].get(iid, "x").strip()


# ---------------------------------------------------------------------------
# Instance table
# ---------------------------------------------------------------------------


def build_table(run_id: str):
    """One row per instance with everything the metrics need."""
    ids = json.loads((ROOT / "data" / "instances.json").read_text())["instance_ids"]
    rows, dropped = [], []
    for iid in ids:
        first = load_log("first", iid)
        if not first or not first.get("attempts"):
            continue  # not run
        row = {"instance_id": iid, "repo": first.get("repo")}
        reports = {s: harness_report(run_id, s, iid) for s in SETS}
        if all(r is None for r in reports.values()):
            continue  # not evaluated yet
        if all(r is not None and r["infra_failure"] for r in reports.values()):
            dropped.append(iid)  # harness environment failed for all six sets
            continue
        for s in SETS:
            r = reports[s]
            row[f"{s}_resolved"] = r["resolved"] if r else None
            row[f"{s}_applied"] = r["applied"] if r else None
            if r and r.get("harness_error"):
                row[f"{s}_harness_error"] = r["harness_error"]

        # Condition B: first attempt that applied in the study's own check.
        b_idx = first.get("b_submitted_index")
        if b_idx is None:
            b_idx = 0
        row["B_index"] = b_idx
        row["B_resolved"] = row[f"a{b_idx + 1}_resolved"]
        row["B_applied"] = row[f"a{b_idx + 1}_applied"]
        row["pass_at_3"] = any(row[f"a{k}_resolved"] for k in (1, 2, 3) if row[f"a{k}_resolved"] is not None)
        row["B_changed"] = b_idx != 0

        # Tokens, calls and change flags per condition.
        a_tokens = [(a["call"].get("input_tokens") or 0, a["call"].get("output_tokens") or 0) for a in first["attempts"]]
        row["B_input_tokens"] = sum(t[0] for t in a_tokens)
        row["B_output_tokens"] = sum(t[1] for t in a_tokens)
        row["B_test_seconds"] = 0.0
        for cond in ("S", "T", "ST"):
            log = load_log(cond, iid)
            if not log or "submitted_patch" not in log:
                row[f"{cond}_changed"] = None
                continue
            steps = log["steps"]
            row[f"{cond}_changed"] = not patches_equal(log["submitted_patch"], log["a1_patch"])
            row[f"{cond}_changed_call2"] = steps[0]["changed"] if steps else None
            row[f"{cond}_changed_call3"] = steps[1]["changed"] if len(steps) > 1 else None
            row[f"{cond}_input_tokens"] = a_tokens[0][0] + sum(s["call"].get("input_tokens") or 0 for s in steps)
            row[f"{cond}_output_tokens"] = a_tokens[0][1] + sum(s["call"].get("output_tokens") or 0 for s in steps)
            row[f"{cond}_test_seconds"] = round(sum(s.get("feedback", {}).get("seconds_total", 0) for s in steps), 2)
            if cond in ("T", "ST"):
                row[f"{cond}_feedback_had_failure"] = any(feedback_reports_failure(s.get("feedback")) for s in steps)
            if cond in ("S", "ST"):
                row[f"{cond}_review_named_defect"] = None  # filled by the coder
        rows.append(row)
    return rows, dropped


def feedback_reports_failure(fb) -> bool:
    """Did the feedback shown to the model contain a failure signal?"""
    if not fb:
        return False
    pf = fb.get("prompt_fields", {})
    if pf.get("apply_result", "").startswith("FAILED"):
        return True
    if "REPRODUCED" in pf.get("repro_after", "") and "NOT REPRODUCED" not in pf.get("repro_after", ""):
        return True
    t = fb.get("tests")
    if t and (t["exit_code"] != 0):
        return True
    return False


def flip_class(before, after) -> str:
    if before is None or after is None:
        return "unknown"
    if not before and after:
        return "fixed"
    if before and not after:
        return "broken"
    if before and after:
        return "kept-right"
    return "kept-wrong"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar test. b = pairs where first wins, c = second wins.
    Under H0 the discordant pairs split 50/50 (binomial, p = 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(pvalues: list) -> list:
    """Holm step-down correction. Returns adjusted p-values in input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvalues[i])
        running = max(running, val)
        adjusted[i] = running
    return adjusted


def bootstrap_ci(values: list, stat, n=BOOT_N, seed=SEED):
    """95 percent percentile bootstrap of stat(sample) over instances."""
    rng = random.Random(seed)
    k = len(values)
    if k == 0:
        return [None, None]
    draws = []
    for _ in range(n):
        sample = [values[rng.randrange(k)] for _ in range(k)]
        draws.append(stat(sample))
    draws.sort()
    return [draws[int(0.025 * n)], draws[int(0.975 * n) - 1]]


def rate(xs):
    return sum(xs) / len(xs) if xs else None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute(rows: list) -> dict:
    conds = ["B", "S", "T", "ST"]
    n = len(rows)
    out = {"n_instances": n, "conditions": {}, "comparisons": [], "pass_at_3": None}

    for c in conds:
        res = [r[f"{c}_resolved"] for r in rows if r.get(f"{c}_resolved") is not None]
        applied = [r[f"{c}_applied"] for r in rows if r.get(f"{c}_applied") is not None]
        flips = {k: 0 for k in ("fixed", "broken", "kept-right", "kept-wrong", "unknown")}
        flip_list = []
        for r in rows:
            fc = flip_class(r.get("a1_resolved"), r.get(f"{c}_resolved"))
            flips[fc] += 1
            flip_list.append(fc)
        changed = [r[f"{c}_changed"] for r in rows if r.get(f"{c}_changed") is not None]
        entry = {
            "n": len(res),
            "resolved": sum(res),
            "resolve_rate": rate(res),
            "resolve_rate_ci95": bootstrap_ci(res, rate),
            "apply_rate": rate(applied),
            "flips_vs_a1": flips,
            "broken_share": flips["broken"] / n if n else None,
            "broken_share_ci95": bootstrap_ci(flip_list, lambda s: s.count("broken") / len(s)),
            "fixed_share": flips["fixed"] / n if n else None,
            "fixed_share_ci95": bootstrap_ci(flip_list, lambda s: s.count("fixed") / len(s)),
            "change_rate": rate(changed),
            "mean_input_tokens": rate([r.get(f"{c}_input_tokens") or 0 for r in rows]),
            "mean_output_tokens": rate([r.get(f"{c}_output_tokens") or 0 for r in rows]),
            "mean_test_seconds": rate([r.get(f"{c}_test_seconds") or 0 for r in rows]),
            "calls_per_instance": 3,
        }
        if c in ("T", "ST"):
            entry["share_with_failure_in_feedback"] = rate([bool(r.get(f"{c}_feedback_had_failure")) for r in rows])
        out["conditions"][c] = entry

    a1 = [r["a1_resolved"] for r in rows if r.get("a1_resolved") is not None]
    out["a1"] = {"resolve_rate": rate(a1), "resolved": sum(a1), "n": len(a1)}
    p3 = [r["pass_at_3"] for r in rows]
    out["pass_at_3"] = {"rate": rate(p3), "count": sum(p3), "note": "oracle upper bound; needs the hidden tests to pick"}

    pvals = []
    for x, y, sq in COMPARISONS:
        pairs = [(r[f"{x}_resolved"], r[f"{y}_resolved"]) for r in rows
                 if r.get(f"{x}_resolved") is not None and r.get(f"{y}_resolved") is not None]
        b = sum(1 for p, q in pairs if p and not q)   # x resolved, y not
        c = sum(1 for p, q in pairs if q and not p)   # y resolved, x not
        diffs = [int(p) - int(q) for p, q in pairs]
        out["comparisons"].append({
            "comparison": f"{x} vs {y}", "sub_question": sq, "n_pairs": len(pairs),
            f"only_{x}_resolved": b, f"only_{y}_resolved": c, "discordant": b + c,
            "rate_difference": rate(diffs),
            "rate_difference_ci95": bootstrap_ci(diffs, rate),
            "mcnemar_p": mcnemar_exact(b, c),
        })
        pvals.append(out["comparisons"][-1]["mcnemar_p"])
    for comp, adj in zip(out["comparisons"], holm(pvals)):
        comp["mcnemar_p_holm"] = adj
        comp["significant_at_0.05"] = adj < ALPHA
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def fmt(x, pct=True):
    if x is None:
        return "n/a"
    if isinstance(x, list):
        return f"[{fmt(x[0], pct)}, {fmt(x[1], pct)}]"
    if isinstance(x, float):
        return f"{100 * x:.1f}" if pct else f"{x:.3g}"
    return str(x)


def write_markdown(res: dict, dropped: list, path: Path):
    L = [f"# Results ({res['n_instances']} instances)", ""]
    L += ["| Condition | Resolved | Resolve rate (%) | 95% CI | Apply rate (%) | Change rate (%) | Fixed | Broken | Kept-right | Kept-wrong | Mean in tokens | Mean out tokens | Mean test s |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c, e in res["conditions"].items():
        f = e["flips_vs_a1"]
        L.append(f"| {c} | {e['resolved']}/{e['n']} | {fmt(e['resolve_rate'])} | {fmt(e['resolve_rate_ci95'])} | {fmt(e['apply_rate'])} | {fmt(e['change_rate'])} | {f['fixed']} | {f['broken']} | {f['kept-right']} | {f['kept-wrong']} | {fmt(e['mean_input_tokens'], False)} | {fmt(e['mean_output_tokens'], False)} | {fmt(e['mean_test_seconds'], False)} |")
    L += ["", f"a1 alone: {res['a1']['resolved']}/{res['a1']['n']} resolved ({fmt(res['a1']['resolve_rate'])} %).",
          f"pass@3 (oracle upper bound): {res['pass_at_3']['count']}/{res['n_instances']} ({fmt(res['pass_at_3']['rate'])} %).", ""]
    L += ["| Comparison | SQ | Pairs | Only first | Only second | Rate diff (pp) | 95% CI | McNemar p | Holm p | Sig. |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in res["comparisons"]:
        x, y = c["comparison"].split(" vs ")
        L.append(f"| {c['comparison']} | {c['sub_question']} | {c['n_pairs']} | {c[f'only_{x}_resolved']} | {c[f'only_{y}_resolved']} | {fmt(c['rate_difference'])} | {fmt(c['rate_difference_ci95'])} | {fmt(c['mcnemar_p'], False)} | {fmt(c['mcnemar_p_holm'], False)} | {'yes' if c['significant_at_0.05'] else 'no'} |")
    if dropped:
        L += ["", f"Dropped (harness environment failed for all six sets): {', '.join(dropped)}"]
    path.write_text("\n".join(L) + "\n")


def write_coding_csv(rows: list, path: Path):
    """One row per (condition, instance) where the submitted patch differs from a1.
    The coder fills edit_type, feedback_had_failure (checked), review_named_defect."""
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "instance_id", "outcome_flip", "edit_type",
                    "feedback_had_failure", "review_named_defect", "note", "coder", "date"])
        for r in rows:
            for c in ("S", "T", "ST"):
                if r.get(f"{c}_changed"):
                    w.writerow([c, r["instance_id"], flip_class(r.get("a1_resolved"), r.get(f"{c}_resolved")),
                                "", r.get(f"{c}_feedback_had_failure", "") if c != "S" else "n/a",
                                "" if c != "T" else "n/a", "", "", ""])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=RUN_ID)
    args = ap.parse_args()
    rows, dropped = build_table(args.run_id)
    if not rows:
        raise SystemExit("no evaluated instances found; run the harness first")
    res = compute(rows)
    res["dropped_instances"] = dropped
    res["run_id"] = args.run_id
    res["instances"] = rows
    (ROOT / "results.json").write_text(json.dumps(res, indent=2))
    write_markdown(res, dropped, ROOT / "results.md")
    write_coding_csv(rows, ROOT / "coding.csv")
    print((ROOT / "results.md").read_text())
    print("wrote results.json, results.md, coding.csv")


if __name__ == "__main__":
    main()
