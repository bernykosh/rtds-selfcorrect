"""Shared helpers for the self-correction study.

Everything that more than one script needs lives here:
- paths and constants from the protocol sheet
- loading the SWE-bench Lite instances
- filling prompt files
- extracting <patch>, <repro> and <review> blocks from model answers
- calling the model with retries and logging
- running commands inside a SWE-bench Docker container
"""

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants from the protocol sheet (Phase 2 doc)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
PROMPT_DIR = ROOT / "prompts"
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "logs"
PRED_DIR = ROOT / "predictions"

INSTANCES_FILE = DATA_DIR / "instances.json"
ORACLE_DIR = DATA_DIR / "oracle_files"
RUN_CONFIG_FILE = ROOT / "run_config.json"

# The swebench 5.x harness loads this dataset name by default. The Phase 2
# protocol named princeton-nlp/SWE-bench_Lite; the SWE-bench organisation now
# hosts the same 300 instances with the extra "image" field the harness needs.
DATASET_NAME = os.environ.get("SWEBENCH_DATASET", "SWE-bench/SWE-bench_Lite")
DATASET_SPLIT = "test"

SEED = 2026
SAMPLE_SIZE = 100
PILOT_SIZE = 50

TEMPERATURE = 0.7
MAX_OUTPUT_TOKENS = 4096
CALLS_PER_CONDITION = 3

FEEDBACK_MAX_LINES = 60
FEEDBACK_MAX_LINE_CHARS = 200
TEST_TIMEOUT_S = 300
REPRO_TIMEOUT_S = 60
SETUP_TIMEOUT_S = 1800  # pip install -e under emulation is slow

MODEL_NAME = os.environ.get("STUDY_MODEL", "Qwen/Qwen3-Coder-480B-A35B-Instruct")
API_BASE_URL = os.environ.get("DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai")
API_KEY_ENV = "DEEPINFRA_API_KEY"

# SWE-bench images are built for amd64 only; on Apple Silicon Docker must be told so.
PLATFORM = os.environ.get("STUDY_DOCKER_PLATFORM", "linux/amd64")

MAX_RETRIES = 5
RETRY_WAIT_S = 10


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def load_lite_dataset(instance_ids=None):
    """Load SWE-bench Lite through the swebench package.

    Returns a list of instance dicts. The swebench loader accepts both the
    Hugging Face name and a local .json/.jsonl path, so a cached copy can be
    pointed to with SWEBENCH_DATASET=path/to/file.json.
    """
    from swebench.harness.utils import load_swebench_dataset

    return load_swebench_dataset(DATASET_NAME, DATASET_SPLIT, instance_ids)


def load_sampled_instances(limit=None):
    """Return the sampled instances in sampled order, with dataset fields."""
    if not INSTANCES_FILE.exists():
        raise SystemExit(f"{INSTANCES_FILE} not found. Run sample_instances.py first.")
    sample = json.loads(INSTANCES_FILE.read_text())
    ids = sample["instance_ids"]
    if limit:
        ids = ids[:limit]
    rows = load_lite_dataset(ids)
    by_id = {r["instance_id"]: r for r in rows}
    return [by_id[i] for i in ids]


def oracle_file_path(instance) -> str:
    """Return the single file that the reference patch changes.

    SWE-bench Lite instances edit exactly one file. The path is read from the
    gold patch header. The gold patch content itself is never shown to the model.
    """
    paths = re.findall(r"^diff --git a/(\S+) b/\S+", instance["patch"], flags=re.M)
    if not paths:
        raise ValueError(f"no file path in gold patch for {instance['instance_id']}")
    if len(set(paths)) > 1:
        # Should not happen in Lite. Record it rather than hide it.
        print(f"WARNING {instance['instance_id']}: gold patch touches {len(set(paths))} files; using the first")
    return paths[0]


def read_oracle_file(instance_id: str) -> str:
    """Oracle file content fetched earlier by fetch_oracle_files.py."""
    f = ORACLE_DIR / f"{instance_id}.txt"
    if not f.exists():
        raise SystemExit(f"{f} not found. Run fetch_oracle_files.py first.")
    return f.read_text()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text()


def fill_prompt(template: str, **fields) -> str:
    """Fill {field} placeholders. str.format would break on braces in code,
    so each placeholder is replaced one by one."""
    out = template
    for key, value in fields.items():
        out = out.replace("{" + key + "}", value if value is not None else "")
    leftover = re.findall(r"\{[a-z_]+\}", out)
    # Placeholders that look like code braces are fine; only our known names matter.
    known = {"{repo}", "{base_commit}", "{problem_statement}", "{file_path}",
             "{file_content}", "{previous_patch}", "{apply_result}",
             "{repro_before}", "{repro_after}", "{test_output}"}
    missing = [x for x in leftover if x in known]
    if missing:
        raise ValueError(f"prompt placeholders not filled: {missing}")
    return out


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------


def _strip_fences(text: str) -> str:
    """Remove ``` fences if the model wrapped a block in them."""
    text = text.strip("\n")
    lines = text.split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def extract_block(answer: str, tag: str) -> str:
    """Return the content of the first <tag>...</tag> block, or '' if absent.

    A missing or unreadable block counts as empty (protocol rule).
    """
    m = re.search(rf"<{tag}>(.*?)</{tag}>", answer, flags=re.S)
    if not m:
        # Model may have opened the tag and never closed it. Take the rest.
        m = re.search(rf"<{tag}>(.*)$", answer, flags=re.S)
        if not m:
            return ""
    return _strip_fences(m.group(1))


def extract_patch(answer: str) -> str:
    patch = extract_block(answer, "patch")
    if not patch.strip():
        return ""
    if not patch.endswith("\n"):
        patch += "\n"
    return patch


def extract_patch_repaired(answer: str) -> str:
    """Extracted patch after hunk-header repair. Used for every patch the
    study applies or shows to the model, in every condition."""
    return repair_hunk_headers(extract_patch(answer))


def extract_repro(answer: str) -> str:
    return extract_block(answer, "repro")


def extract_review(answer: str) -> str:
    return extract_block(answer, "review")


def patches_equal(a: str, b: str) -> bool:
    """Two patches count as the same when they match after trailing
    whitespace on each line is removed."""
    norm = lambda s: "\n".join(line.rstrip() for line in (s or "").strip().split("\n"))
    return norm(a) == norm(b)


# ---------------------------------------------------------------------------
# Model calls
# ---------------------------------------------------------------------------

_client = None


def get_client():
    global _client
    if _client is None:
        from openai import OpenAI

        key = os.environ.get(API_KEY_ENV)
        if not key:
            raise SystemExit(f"Environment variable {API_KEY_ENV} is not set.")
        _client = OpenAI(api_key=key, base_url=API_BASE_URL)
    return _client


def call_model(prompt: str, dry_run: bool = False) -> dict:
    """One model call. Returns a dict that goes straight into the log.

    Retries only on transport or server errors, with the same input. Each
    retry is recorded in the returned dict. The call still counts as one call.
    """
    record = {
        "model": MODEL_NAME,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "prompt": prompt,
        "started_at": now_iso(),
        "retries": [],
        "answer": "",
        "finish_reason": None,
        "input_tokens": None,
        "output_tokens": None,
    }
    if dry_run:
        record["answer"] = "<review>dry run</review>\n<patch>\n</patch>\n<repro>\nprint('NOT REPRODUCED')\n</repro>"
        record["finished_at"] = now_iso()
        return record

    client = get_client()
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_OUTPUT_TOKENS,
            )
            choice = resp.choices[0]
            record["answer"] = choice.message.content or ""
            record["finish_reason"] = choice.finish_reason
            if resp.usage:
                record["input_tokens"] = resp.usage.prompt_tokens
                record["output_tokens"] = resp.usage.completion_tokens
            record["finished_at"] = now_iso()
            return record
        except Exception as e:  # network, rate limit, 5xx
            record["retries"].append({"at": now_iso(), "error": f"{type(e).__name__}: {e}"[:500]})
            if attempt == MAX_RETRIES:
                record["finished_at"] = now_iso()
                record["error"] = "gave up after retries"
                return record
            time.sleep(RETRY_WAIT_S * (attempt + 1))


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------


def activation_line(instance) -> str:
    """Build the environment activation for feedback commands from the
    instance's own eval script, so that they run in the same Python
    environment as the hidden tests. SWE-bench eval scripts activate in two
    steps (source .../activate, then conda activate testbed); both are kept.

    The lines are joined with ";" and each one is silenced on its own. They
    must not run in a subshell, because a subshell cannot change the
    environment of the command that follows.
    """
    lines = []
    for line in instance.get("eval_script", "").split("\n"):
        t = line.strip()
        if (t.startswith("source") and "activate" in t) or t.startswith("conda activate"):
            if t not in lines:
                lines.append(t)
    if not lines:
        lines = ["source /opt/miniconda3/bin/activate", "conda activate testbed"]
    return "; ".join(f"{l} >/dev/null 2>&1" for l in lines)


def setup_commands(instance) -> list:
    """The install commands the instance's eval script runs before its tests.

    SWE-bench images ship the repository but not its installed package: the
    eval script installs it (for example "python -m pip install -e .[test]"),
    which also brings in pytest. The feedback step needs the same environment,
    so the same commands are run once per container.

    Only lines before the test output marker are read. Lines that touch test
    content are never taken: the test patch (a git apply heredoc), any git
    checkout of a test file, and the test command itself. This keeps the
    hidden tests out of the feedback step.
    """
    script = instance.get("eval_script", "")
    cmds, in_heredoc = [], False
    for line in script.split("\n"):
        t = line.strip()
        if ">>>>> Start Test Output" in t:
            break
        if in_heredoc:
            if t.startswith("EOF_"):
                in_heredoc = False
            continue
        if t.startswith("git apply") and "<<" in t:
            in_heredoc = True
            continue
        if not t or t.startswith("#") or t.startswith(("set ", "source", "conda activate", ": '")):
            continue
        if t.startswith(("git checkout", "git status", "git show", "git diff", "git -c")):
            continue
        if t.startswith("cd "):
            continue
        cmds.append(t)
    return cmds


class Container:
    """A running container built from the instance's SWE-bench image.

    Usage:
        with Container(instance) as c:
            out = c.run("git status")
    """

    def __init__(self, instance, name_suffix="study"):
        import docker

        self.instance = instance
        self.image = instance["image"]
        self.client = docker.from_env()
        self.name = f"sweb.study.{instance['instance_id'].lower()}.{name_suffix}"
        self.container = None
        self.activate = activation_line(instance)
        self.setup_cmds = setup_commands(instance)
        self.prepared = False

    def __enter__(self):
        import docker

        try:
            self.client.images.get(self.image)
        except docker.errors.ImageNotFound:
            print(f"  pulling {self.image} ({PLATFORM}) ...")
            self.client.images.pull(self.image, platform=PLATFORM)
            self.client.images.get(self.image)  # raises a clear error if the pull failed
        try:
            old = self.client.containers.get(self.name)
            old.remove(force=True)
        except docker.errors.NotFound:
            pass
        self.container = self.client.containers.run(
            self.image, command="tail -f /dev/null", name=self.name,
            detach=True, user="root", working_dir="/testbed", platform=PLATFORM,
        )
        return self

    def __exit__(self, *exc):
        if self.container:
            try:
                self.container.remove(force=True)
            except Exception:
                pass

    def run(self, cmd: str, timeout=None) -> dict:
        """Run a shell command in /testbed inside the activated environment."""
        full = f"{self.activate}; cd /testbed; {cmd}"
        if timeout:
            full = f"timeout {timeout} bash -c {_shell_quote(full)}"
        t0 = time.time()
        res = self.container.exec_run(["/bin/bash", "-c", full], user="root", workdir="/testbed")
        return {
            "cmd": cmd,
            "exit_code": res.exit_code,
            "output": res.output.decode("utf-8", errors="replace"),
            "seconds": round(time.time() - t0, 2),
            "timed_out": timeout is not None and res.exit_code == 124,
        }

    def prepare(self) -> list:
        """Run the instance's own install commands once in this container.

        Without this the repository package and pytest are missing, and every
        feedback command fails for a reason that has nothing to do with the
        patch. The results are returned so that the log records them.
        """
        if self.prepared:
            return []
        results = []
        for cmd in self.setup_cmds:
            r = self.run(cmd, timeout=SETUP_TIMEOUT_S)
            r["output"] = r["output"][-3000:]  # keep the tail; pip is verbose
            results.append(r)
            if r["exit_code"] != 0:
                print(f"  WARNING setup command failed ({r['exit_code']}): {cmd}")
        self.prepared = True
        return results

    def write_file(self, path: str, content: str):
        """Write a text file into the container."""
        import io
        import tarfile

        data = content.encode("utf-8")
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as tar:
            info = tarfile.TarInfo(name=os.path.basename(path))
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        stream.seek(0)
        self.container.put_archive(os.path.dirname(path) or "/", stream.read())

    def read_file(self, path: str) -> str:
        res = self.container.exec_run(["cat", path])
        if res.exit_code != 0:
            raise FileNotFoundError(path)
        return res.output.decode("utf-8", errors="replace")

    def reset(self):
        """Return the working tree to base_commit."""
        self.run("git checkout -- . && git clean -fdq")

    def apply_patch(self, patch: str) -> dict:
        """Apply a patch with the same first command the harness uses."""
        if not patch.strip():
            return {"applied": False, "output": "empty patch"}
        self.write_file("/tmp/study_patch.diff", patch)
        res = self.run("git apply --verbose /tmp/study_patch.diff")
        return {"applied": res["exit_code"] == 0, "output": res["output"][-2000:]}

    def check_patch(self, patch: str) -> bool:
        """git apply --check, without changing the tree."""
        if not patch.strip():
            return False
        self.write_file("/tmp/study_patch.diff", patch)
        res = self.run("git apply --check /tmp/study_patch.diff")
        return res["exit_code"] == 0


def _shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------


def log_path(condition: str, instance_id: str) -> Path:
    d = LOG_DIR / condition
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{instance_id}.json"


def save_log(condition: str, instance_id: str, data: dict):
    data.setdefault("instance_id", instance_id)
    data.setdefault("condition", condition)
    data["saved_at"] = now_iso()
    log_path(condition, instance_id).write_text(json.dumps(data, indent=2))


def load_log(condition: str, instance_id: str):
    p = log_path(condition, instance_id)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def write_run_config(extra=None):
    cfg = {
        "model": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "temperature": TEMPERATURE,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "calls_per_instance_per_condition": CALLS_PER_CONDITION,
        "dataset": DATASET_NAME,
        "split": DATASET_SPLIT,
        "seed": SEED,
        "sample_size": SAMPLE_SIZE,
        "pilot_size": PILOT_SIZE,
        "feedback_max_lines": FEEDBACK_MAX_LINES,
        "feedback_max_line_chars": FEEDBACK_MAX_LINE_CHARS,
        "test_timeout_s": TEST_TIMEOUT_S,
        "repro_timeout_s": REPRO_TIMEOUT_S,
        "setup_timeout_s": SETUP_TIMEOUT_S,
        "written_at": now_iso(),
    }
    if RUN_CONFIG_FILE.exists():
        old = json.loads(RUN_CONFIG_FILE.read_text())
        cfg["first_written_at"] = old.get("first_written_at", old.get("written_at"))
    else:
        cfg["first_written_at"] = cfg["written_at"]
    if extra:
        cfg.update(extra)
    try:
        import importlib.metadata as m
        cfg["swebench_version"] = m.version("swebench")
    except Exception:
        pass
    RUN_CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
    return cfg


# ---------------------------------------------------------------------------
# Patch format repair
# ---------------------------------------------------------------------------


def repair_hunk_headers(patch: str) -> str:
    """Recompute the line counts in every @@ hunk header from the hunk body.

    Models often write a correct diff body with wrong counts in the header
    (for example "@@ -10,7 +10,8 @@" when the body has 9 old lines). git
    rejects such a patch as corrupt without looking at the content. This
    function rewrites only the counts; the start line of the old file is kept
    and the start line of the new file is derived from the earlier hunks.
    A blank line inside a hunk body is treated as an empty context line.
    The content of the change is never altered.
    """
    import re

    if not patch.strip():
        return patch
    lines = patch.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    out = []
    i = 0
    offset = 0  # new_start - old_start carried over from earlier hunks
    hunk_re = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@(.*)$")
    while i < len(lines):
        m = hunk_re.match(lines[i])
        if not m:
            if lines[i].startswith(("diff ", "--- ", "+++ ", "index ")):
                offset = 0 if lines[i].startswith("diff ") else offset
            out.append(lines[i])
            i += 1
            continue
        old_start = int(m.group(1))
        tail = m.group(3)
        body = []
        i += 1
        while i < len(lines) and not hunk_re.match(lines[i]) and not lines[i].startswith(("diff ", "--- a/", "+++ b/")):
            body.append(lines[i])
            i += 1
        # Drop trailing blank lines that belong to the answer, not the hunk.
        while body and body[-1] == "":
            body.pop()
        fixed_body = []
        old_n = new_n = 0
        for b in body:
            if b.startswith("\\"):
                fixed_body.append(b)
                continue
            if b == "":
                b = " "
            tag = b[0]
            if tag == " ":
                old_n += 1
                new_n += 1
            elif tag == "-":
                old_n += 1
            elif tag == "+":
                new_n += 1
            else:  # a context line that lost its leading space
                b = " " + b
                old_n += 1
                new_n += 1
            fixed_body.append(b)
        new_start = old_start + offset
        out.append(f"@@ -{old_start},{old_n} +{new_start},{new_n} @@{tail}")
        out.extend(fixed_body)
        offset += new_n - old_n
    return "\n".join(out) + "\n"
