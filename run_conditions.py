"""Steps 3 to 6 of the protocol: run the model under each condition.

    python run_conditions.py first  [--limit 50]   # a1, a2, a3 with P1
    python run_conditions.py S      [--limit 50]   # a1 -> two self-review calls
    python run_conditions.py T      [--limit 50]   # a1 -> two execution-feedback calls
    python run_conditions.py ST     [--limit 50]   # a1 -> two calls with both

Condition B (budget-matched baseline) needs no model calls of its own: it
submits the first of a1, a2, a3 that applies cleanly. The "first" run does the
apply check at its end and stores it in the same log.

Every condition makes exactly three model calls per instance. S, T and ST
reuse a1 from the "first" log as their first call, so they make two calls here.

One JSON log per instance per condition goes to logs/<condition>/<id>.json.
A run skips instances with a complete log, so you can resume it.
"""

import argparse
import sys

from common import (
    CALLS_PER_CONDITION,
    FEEDBACK_MAX_LINE_CHARS,
    FEEDBACK_MAX_LINES,
    REPRO_TIMEOUT_S,
    TEST_TIMEOUT_S,
    Container,
    call_model,
    extract_patch,
    extract_patch_repaired,
    extract_repro,
    extract_review,
    fill_prompt,
    load_log,
    load_prompt,
    load_sampled_instances,
    oracle_file_path,
    patches_equal,
    read_oracle_file,
    save_log,
    write_run_config,
)

PROMPT_FILES = {
    "first": "P1_generate.txt",
    "S": "P2_S_selfreview.txt",
    "T": "P2_T_feedback.txt",
    "ST": "P2_ST_both.txt",
}


def base_fields(inst):
    return dict(
        repo=inst["repo"],
        base_commit=inst["base_commit"],
        problem_statement=inst["problem_statement"],
        file_path=oracle_file_path(inst),
        file_content=read_oracle_file(inst["instance_id"]),
    )


# ---------------------------------------------------------------------------
# Condition "first": three independent attempts, then apply check for B
# ---------------------------------------------------------------------------


def run_first(inst, dry_run, skip_apply_check):
    iid = inst["instance_id"]
    template = load_prompt(PROMPT_FILES["first"])
    prompt = fill_prompt(template, **base_fields(inst))
    attempts = []
    for k in range(CALLS_PER_CONDITION):
        rec = call_model(prompt, dry_run)
        attempts.append({
            "attempt": k + 1,
            "call": rec,
            "patch_raw": extract_patch(rec["answer"]),
            "patch": extract_patch_repaired(rec["answer"]),
            "repro": extract_repro(rec["answer"]),
            "applies": None,
        })
        print(f"  a{k + 1}: {rec.get('output_tokens')} output tokens, patch {'empty' if not attempts[-1]['patch'] else 'ok'}")

    log = {
        "instance_id": iid,
        "condition": "first",
        "repo": inst["repo"],
        "oracle_file": oracle_file_path(inst),
        "attempts": attempts,
        "b_submitted_index": None,
        "b_none_applied": None,
    }
    if not skip_apply_check and not dry_run:
        apply_check(inst, log)
    return log


def apply_check(inst, log):
    """git apply --check for a1, a2, a3 in the instance container. Picks B."""
    attempts = log["attempts"]
    with Container(inst, name_suffix="apply") as c:
        for a in attempts:
            a["applies"] = c.check_patch(a["patch"])
    first_ok = next((i for i, a in enumerate(attempts) if a["applies"]), None)
    log["b_submitted_index"] = first_ok if first_ok is not None else 0
    log["b_none_applied"] = first_ok is None
    print(f"  applies: {[a['applies'] for a in attempts]} -> B submits a{log['b_submitted_index'] + 1}")


def reextract_first(inst, dry_run):
    """Re-read the stored answers of condition first, extract the patches
    again with hunk-header repair, and redo the apply check. No model call."""
    iid = inst["instance_id"]
    log = load_log("first", iid)
    if log is None or not log.get("attempts"):
        raise SystemExit(f"logs/first/{iid}.json missing")
    for a in log["attempts"]:
        a["patch_raw"] = extract_patch(a["call"]["answer"])
        a["patch"] = extract_patch_repaired(a["call"]["answer"])
    log["reextracted_with_hunk_repair"] = True
    if not dry_run:
        apply_check(inst, log)
    return log


# ---------------------------------------------------------------------------
# Execution feedback for T and ST
# ---------------------------------------------------------------------------


def cut_output(text: str) -> str:
    lines = text.replace("\r", "").split("\n")[:FEEDBACK_MAX_LINES]
    return "\n".join(line[:FEEDBACK_MAX_LINE_CHARS] for line in lines)


def summarise_repro(res) -> str:
    """One line for the prompt: exit code and the REPRODUCED verdict if printed."""
    if res is None:
        return "no reproduction script"
    out = res["output"]
    verdict = "no verdict printed"
    for line in reversed(out.strip().split("\n")):
        if "NOT REPRODUCED" in line:
            verdict = "printed NOT REPRODUCED"
            break
        if "REPRODUCED" in line:
            verdict = "printed REPRODUCED"
            break
    if res["timed_out"]:
        return f"timed out after {REPRO_TIMEOUT_S} s"
    tail = out.strip().split("\n")[-1][:FEEDBACK_MAX_LINE_CHARS] if out.strip() else ""
    extra = f'; last line: "{tail}"' if verdict == "no verdict printed" and tail else ""
    return f"exit code {res['exit_code']}, {verdict}{extra}"


def find_test_files(c: Container, oracle_path: str) -> list:
    """Test files whose name matches the changed module (protocol rule)."""
    stem = oracle_path.rsplit("/", 1)[-1]
    stem = stem[:-3] if stem.endswith(".py") else stem
    if stem == "__init__":
        stem = oracle_path.rsplit("/", 2)[-2]
    names = f'-name "test_{stem}.py" -o -name "{stem}_test.py" -o -name "test_{stem}s.py"'
    res = c.run(f'find /testbed -not -path "*/.git/*" \\( {names} \\) | sort | head -3')
    return [p.strip() for p in res["output"].split("\n") if p.strip()]


def test_command(inst, files: list) -> str:
    """pytest on the matching files. Django tests need its own runner, so for
    django/django the test labels are the directories of the matching files."""
    if inst["repo"] == "django/django":
        labels = sorted({f.split("/testbed/tests/", 1)[1].split("/")[0] for f in files if "/testbed/tests/" in f})
        if not labels:
            return ""
        return f"python tests/runtests.py --parallel 1 {' '.join(labels)}"
    rel = [f.replace("/testbed/", "", 1) for f in files]
    return f"python -m pytest -q --tb=short {' '.join(rel)}"


def execution_feedback(c: Container, inst, patch: str, repro: str) -> dict:
    """Apply the patch and run the reproduction script and matching tests.

    This never runs the hidden FAIL_TO_PASS and PASS_TO_PASS tests.
    """
    fb = {"seconds_total": 0.0}
    c.reset()
    has_repro = bool(repro.strip())
    if has_repro:
        c.write_file("/tmp/repro.py", repro)
        fb["repro_before"] = c.run("python /tmp/repro.py", timeout=REPRO_TIMEOUT_S)
        fb["seconds_total"] += fb["repro_before"]["seconds"]
    else:
        fb["repro_before"] = None

    applied = c.apply_patch(patch)
    fb["apply"] = applied

    if has_repro:
        fb["repro_after"] = c.run("python /tmp/repro.py", timeout=REPRO_TIMEOUT_S)
        fb["seconds_total"] += fb["repro_after"]["seconds"]
    else:
        fb["repro_after"] = None

    files = find_test_files(c, oracle_file_path(inst))
    fb["test_files"] = files
    cmd = test_command(inst, files)
    fb["test_cmd"] = cmd
    if cmd:
        fb["tests"] = c.run(cmd, timeout=TEST_TIMEOUT_S)
        fb["seconds_total"] += fb["tests"]["seconds"]
        test_output = fb["tests"]["output"]
        if fb["tests"]["timed_out"]:
            test_output = f"(timed out after {TEST_TIMEOUT_S} s)\n" + test_output
    else:
        fb["tests"] = None
        stem = oracle_file_path(inst).rsplit("/", 1)[-1]
        test_output = f"No test file whose name matches {stem} was found in the repository."

    fb["prompt_fields"] = {
        "apply_result": "applied" if applied["applied"] else "FAILED to apply: " + cut_output(applied["output"]).split("\n")[0],
        "repro_before": summarise_repro(fb["repro_before"]),
        "repro_after": summarise_repro(fb["repro_after"]),
        "test_output": cut_output(test_output),
    }
    fb["seconds_total"] = round(fb["seconds_total"], 2)
    return fb


# ---------------------------------------------------------------------------
# Revision conditions S, T, ST
# ---------------------------------------------------------------------------


def run_revision(inst, condition, dry_run):
    iid = inst["instance_id"]
    first = load_log("first", iid)
    if first is None:
        raise SystemExit(f"logs/first/{iid}.json missing; run condition first before {condition}")
    a1 = first["attempts"][0]
    template = load_prompt(PROMPT_FILES[condition])
    fields = base_fields(inst)

    log = {
        "instance_id": iid,
        "condition": condition,
        "repo": inst["repo"],
        "oracle_file": oracle_file_path(inst),
        "a1_patch": a1["patch"],
        "a1_repro": a1["repro"],
        "steps": [],
    }
    previous = a1["patch"]
    uses_feedback = condition in ("T", "ST")

    container = Container(inst, name_suffix=condition.lower()) if (uses_feedback and not dry_run) else None
    if container:
        container.__enter__()
        print("  installing the repository package for the feedback step ...")
        log["setup"] = container.prepare()
        ok = all(r["exit_code"] == 0 for r in log["setup"])
        log["setup_ok"] = ok
        if not ok:
            print("  WARNING: setup did not finish cleanly; the feedback may be unreliable for this instance")
    try:
        for k in range(CALLS_PER_CONDITION - 1):
            step = {"call_number": k + 2, "previous_patch": previous}
            f = dict(fields, previous_patch=previous)
            if uses_feedback:
                if container:
                    fb = execution_feedback(container, inst, previous, a1["repro"])
                else:
                    fb = {"prompt_fields": {"apply_result": "dry", "repro_before": "dry", "repro_after": "dry", "test_output": "dry"}, "seconds_total": 0}
                step["feedback"] = fb
                f.update(fb["prompt_fields"])
            prompt = fill_prompt(template, **f)
            rec = call_model(prompt, dry_run)
            step["call"] = rec
            step["review"] = extract_review(rec["answer"]) if condition in ("S", "ST") else None
            step["patch_raw"] = extract_patch(rec["answer"])
            step["patch"] = extract_patch_repaired(rec["answer"])
            step["changed"] = not patches_equal(step["patch"], previous)
            log["steps"].append(step)
            print(f"  call {k + 2}: {'changed' if step['changed'] else 'unchanged'}, {rec.get('output_tokens')} output tokens"
                  + (f", feedback {fb['seconds_total']} s" if uses_feedback else ""))
            previous = step["patch"]
    finally:
        if container:
            container.__exit__(None, None, None)

    log["submitted_patch"] = previous
    log["changed_from_a1"] = not patches_equal(previous, a1["patch"])
    return log


# ---------------------------------------------------------------------------


def is_complete(log: dict, condition: str) -> bool:
    if log is None:
        return False
    calls = [a["call"] for a in log["attempts"]] if condition == "first" else [s["call"] for s in log["steps"]]
    return len(calls) == (3 if condition == "first" else 2) and not any(c.get("error") for c in calls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("condition", choices=list(PROMPT_FILES))
    ap.add_argument("--limit", type=int, default=None, help="first N sampled instances (pilot = 50)")
    ap.add_argument("--only", nargs="*", default=None, help="specific instance IDs")
    ap.add_argument("--dry-run", action="store_true", help="no API, no Docker; checks prompts and logs")
    ap.add_argument("--skip-apply-check", action="store_true", help="first only: do not check git apply")
    ap.add_argument("--redo", action="store_true", help="overwrite existing logs")
    ap.add_argument("--reextract", action="store_true",
                    help="first only: re-extract patches from stored answers with hunk repair; no model calls")
    args = ap.parse_args()

    write_run_config()
    instances = load_sampled_instances(args.limit)
    if args.only:
        instances = [i for i in instances if i["instance_id"] in set(args.only)]

    done = skipped = failed = 0
    for n, inst in enumerate(instances, 1):
        iid = inst["instance_id"]
        existing = load_log(args.condition, iid)
        if args.reextract:
            if args.condition != "first":
                raise SystemExit("--reextract works with condition first only")
            if existing and existing.get("reextracted_with_hunk_repair") and not args.redo:
                skipped += 1
                continue
        elif existing and is_complete(existing, args.condition) and not args.redo:
            skipped += 1
            continue
        print(f"[{n}/{len(instances)}] {args.condition} {iid}")
        try:
            if args.reextract:
                log = reextract_first(inst, args.dry_run)
            elif args.condition == "first":
                log = run_first(inst, args.dry_run, args.skip_apply_check)
            else:
                log = run_revision(inst, args.condition, args.dry_run)
            save_log(args.condition, iid, log)
            done += 1
        except KeyboardInterrupt:
            print("interrupted; rerun to resume")
            sys.exit(1)
        except Exception as e:
            failed += 1
            print(f"  ERROR {type(e).__name__}: {e}")
            save_log(args.condition, iid, {"error": f"{type(e).__name__}: {e}", "attempts": [], "steps": []})
    print(f"{args.condition}: {done} done, {skipped} skipped (already complete), {failed} failed")


if __name__ == "__main__":
    main()
