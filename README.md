# Self-correction study: run scripts

Code for the RTDS HT2026 research project. The study asks whether intrinsic
self-correction adds value beyond running tests on real code projects, at an
equal compute budget. The design is in the Phase 2 document; this folder
implements its protocol sheet step by step.

## What is in the folder

| File | Protocol step | What it does |
| --- | --- | --- |
| `prompts/` | prompt texts | The four prompt files, exact text from the Phase 2 document |
| `common.py` | all | Shared helpers: dataset loading, prompt filling, patch extraction, model calls, container commands |
| `sample_instances.py` | 2 | Samples 100 instances (seed 2026, stratified by repository) into `data/instances.json` |
| `fetch_oracle_files.py` | model input | Reads the oracle file for each instance from its Docker image into `data/oracle_files/` |
| `run_conditions.py` | 3 to 6 | Runs condition `first` (a1, a2, a3), `S`, `T`, `ST`; one JSON log per instance per condition in `logs/` |
| `build_predictions.py` | 7 | Writes six predictions files (a1, a2, a3, S, T, ST) into `predictions/` |
| `run_eval.sh` | 7 | Runs the official harness on each predictions file |
| `batch.sh` | 2 to 7 | Runs everything for a slice of instances, then removes their images (disk limit) |
| `analyze.py` | 8, 9 | Writes `results.json`, `results.md`, `coding.csv` |
| `run_config.json` | protocol | Written by `run_conditions.py`: model, temperature, limits, swebench version, dates |

## Setup (once)

```bash
cd rtds-selfcorrect
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
docker info            # Docker Desktop must be running
echo $DEEPINFRA_API_KEY | cut -c1-6   # key must be set in the shell
```

Two settings are read from the environment and have defaults:

- `STUDY_MODEL`, default `Qwen/Qwen3-Coder-480B-A35B-Instruct`
- `DEEPINFRA_BASE_URL`, default `https://api.deepinfra.com/v1/openai`

Check both against the DeepInfra model page before the first run. They are
recorded in `run_config.json`.

### One check before anything else

The `swebench` package installed by `requirements.txt` is version 5.0.2. It
loads the dataset `SWE-bench/SWE-bench_Lite` and needs an `image` field on
each instance. Run this once:

```bash
python -c "from common import load_lite_dataset; r=load_lite_dataset()[0]; print(len(load_lite_dataset()), r['instance_id'], r['image'])"
```

Expected: `300`, an instance ID, and an image name. If this fails, the dataset
or harness version has moved; stop and report the error.

## Pilot run (first 50 instances)

```bash
python sample_instances.py                 # writes data/instances.json
python fetch_oracle_files.py --limit 50    # pulls images; needs disk
python run_conditions.py first --limit 50  # 3 model calls per instance + git apply check
python run_conditions.py S --limit 50      # 2 model calls per instance
python run_conditions.py T --limit 50      # 2 model calls + container runs per instance
python run_conditions.py ST --limit 50
python build_predictions.py --limit 50
./run_eval.sh                              # six harness runs
python analyze.py
```

Every script can be stopped and rerun; instances with a complete log are
skipped. Pass `--redo` to overwrite.

### Disk

Each instance image is roughly 1 to 3 GB. With a 40 GB Docker limit, run in
slices and let `batch.sh` remove the images of a finished slice:

```bash
./batch.sh 0 20     # instances 0 to 19 in sampled order
./batch.sh 20 40
./batch.sh 40 50    # end of the pilot
python analyze.py
```

Set `KEEP_IMAGES=1` to keep the images. Logs and harness reports are kept
across slices; `analyze.py` reads them all.

## What the harness produces

- `logs/run_evaluation/study2026/<set>/<instance_id>/report.json` with
  `resolved` and `patch_successfully_applied` for each patch set
- `harness_reports/<set>.study2026.json` with the summary per set

`analyze.py` reads the per-instance reports, so `build_predictions.py` must use
the same set names (`a1`, `a2`, `a3`, `S`, `T`, `ST`) as `model_name_or_path`.

## Rules that the scripts enforce

- Hidden tests (`FAIL_TO_PASS`, `PASS_TO_PASS`) are never read by the study
  scripts. Only the harness in step 7 runs them.
- Exactly three model calls per instance per condition. A call that fails for
  a network reason is retried with the same input; each retry is stored in
  the log under `retries`.
- A patch the extractor cannot read is stored as an empty patch.
- Condition B submits the first of a1, a2, a3 that passes `git apply --check`
  in the instance container. If none applies, B submits a1 and the log marks
  `b_none_applied`.
- Feedback for T and ST: `git apply`, the model's reproduction script before
  and after the patch (60 s timeout), and the repository test files whose
  name matches the changed module (`test_<module>.py`, `<module>_test.py`,
  `test_<module>s.py`; at most three files; 300 s timeout). For
  `django/django` the matching files' test labels are run with
  `tests/runtests.py`, because pytest cannot run Django's test suite. If no
  matching file exists, the feedback says so. The first 60 lines of output,
  each cut at 200 characters, go to the model.

## Things to decide before the full run

These are design points the Phase 2 document left for confirmation. The
scripts implement the first option of each; change the flag or code if you
prefer the other.

1. B selection rule: first attempt that applies (implemented) or a random
   attempt.
2. Test file match rule for feedback: name match as above (implemented). For
   Django instances this often finds nothing, so T gets only the
   reproduction-script signal there. Record this as a limitation.
3. Model: `Qwen/Qwen3-Coder-480B-A35B-Instruct` on DeepInfra (open weights,
   hosted). Confirm the exact model string on the DeepInfra page.

## Logs to return

After the pilot, return:

- `data/instances.json`, `data/oracle_paths.json`, `run_config.json`
- `logs/first`, `logs/S`, `logs/T`, `logs/ST` (model logs)
- `logs/run_evaluation/study2026/` (harness logs; `report.json` files are enough)
- `harness_reports/`, `results.json`, `results.md`, `coding.csv`

`logs/` contains the full prompts, so it is large. Zip it.
