"""SWE-Gym data/evaluator preparation adapted from Accio-Lab/Dressage.

Source: examples/data/swegym/prepare_swegym_data.py
Upstream commit: 3e3142fe8ea07e4504c3b20a936a4c201a3de44c
License: Apache-2.0; see LICENSE.dressage.
Changes: extract only dataset, sanitizer and official evaluator preparation;
rename reward marker and temporary paths for Miles. No Dressage runtime dependency.
"""

from __future__ import annotations

import base64
import json
import re
import shlex
import sys
import types
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

DATASET_NAME = "NovaSky-AI/SkyRL-v0-293-data"

SWEGYM_HARNESS_COMMIT = "16dd480cce9b27bf111a362d280881c6def5d2a7"

EXPECTED_SPLIT_SIZES = {"train": 293, "validation": 23}

REWARD_MARKER = "MILES_SWEGYM_REWARD_JSON="

GIT_SANITIZE_CMD_NAME = "swegym_git_sanitize"

DEFAULT_GIT_SANITIZE_TIMEOUT = 120

LEGACY_SWEBENCH_IMAGE_REPOS = {
    "marshmallow-code/marshmallow",
    "pydicom/pydicom",
    "pylint-dev/astroid",
    "pvlib/pvlib-python",
    "pyvista/pyvista",
    "sqlfluff/sqlfluff",
}

TEST_STATUS_IMPORT = "from swegym.harness.constants import TestStatus"

STANDALONE_TEST_STATUS = '''class TestStatus(Enum):
    FAILED = "FAILED"
    PASSED = "PASSED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"
    XFAIL = "XFAIL"'''

SANDBOX_GRADING_DRIVER = r"""
import base64,json,sys,zlib

log_path,expected_b64,repo,eval_rc=sys.argv[1:]
content=open(log_path,encoding="utf-8",errors="replace").read()
expected=json.loads(zlib.decompress(base64.b64decode(expected_b64)).decode())
repo_key=repo.lower()
parser=MAP_REPO_TO_PARSER.get(repo_key)
if parser is None:
    statuses={}
    parser_error="missing_official_log_parser"
else:
    try:
        statuses=parser(content)
        parser_error=""
    except Exception as exc:
        statuses={}
        parser_error="official_log_parser_exception:"+type(exc).__name__+":"+str(exc)

passing={"PASSED","XFAIL"}
f2p=list(expected.get("FAIL_TO_PASS") or [])
p2p=list(expected.get("PASS_TO_PASS") or [])
f2p_ok=[case for case in f2p if statuses.get(case) in passing]
p2p_ok=[case for case in p2p if statuses.get(case) in passing]
resolved=len(f2p_ok)==len(f2p) and len(p2p_ok)==len(p2p)
diag={
    "resolved":resolved,
    "fail_to_pass_success":len(f2p_ok),
    "fail_to_pass_total":len(f2p),
    "pass_to_pass_success":len(p2p_ok),
    "pass_to_pass_total":len(p2p),
    "parsed_tests":len(statuses),
    "eval_returncode":int(eval_rc),
    "harness":"swegym",
    "log_parser_repo":repo_key,
}
if not resolved:
    if parser_error:
        diag["failure_reason"]=parser_error
    elif not statuses:
        diag["failure_reason"]="test_log_parse_failure"
    elif len(f2p_ok)!=len(f2p):
        diag["failure_reason"]="fail_to_pass_failure"
    else:
        diag["failure_reason"]="pass_to_pass_regression"
if not statuses:
    diag["test_log_tail"]=content[-4000:]
print("MILES_SWEGYM_REWARD_JSON="+json.dumps(diag,separators=(",",":")))
"""


def standalone_log_parser_source(official_source: str) -> str:
    """Make the pinned SWE-Gym log parser executable inside task images."""

    if TEST_STATUS_IMPORT not in official_source:
        raise RuntimeError("unexpected SWE-Gym log_parsers.py: TestStatus import missing")
    standalone = official_source.replace(TEST_STATUS_IMPORT, STANDALONE_TEST_STATUS, 1)
    if "MAP_REPO_TO_PARSER" not in standalone:
        raise RuntimeError("unexpected SWE-Gym log_parsers.py: parser map missing")
    # Several SWE-Gym images still use Python 3.7/3.8, while the pinned parser
    # annotates functions with PEP 585 built-in generics such as dict[str, str].
    # Postponing annotation evaluation keeps the exact parser implementation
    # executable in those older task environments.
    if not standalone.startswith("from __future__ import annotations\n"):
        standalone = "from __future__ import annotations\n" + standalone
    return standalone.rstrip() + "\n\n" + SANDBOX_GRADING_DRIVER


def format_prompt(problem_statement: str) -> str:
    instruction = (
        "Fix the repository issue below by editing the source files in /testbed. "
        "Inspect the relevant code, implement the fix directly in the working tree, "
        "and run focused existing tests. The testbed Conda environment is already active. "
        "Use focused searches to locate the implementation, then Read with an offset and a limit of at most 150 lines. "
        "Limit Grep output to 40 lines and avoid reading entire large files. "
        "Implement a minimal fix promptly rather than spending the entire budget exploring.\n\nIssue:\n"
    )
    return instruction + problem_statement.strip() + "\n\nIntegrity rules:\n- Do not modify tests, pytest configuration, or evaluation harness files.\n- Do not retrieve a solution, patch, commit, or pull request from the task repository or its mirrors."


def registry_image_for_instance(instance_id: str, repo: str) -> str:
    """Return the image naming convention used by the reference run."""

    owner, repo_with_issue = instance_id.split("__", 1)
    image_repo, issue_id = repo_with_issue.rsplit("-", 1)
    if repo in LEGACY_SWEBENCH_IMAGE_REPOS:
        return f"swebench/sweb.eval.x86_64.{owner}_1776_{image_repo}-{issue_id}:latest"
    suffix = instance_id.replace("__", "_s_").lower()
    return f"xingyaoww/sweb.eval.x86_64.{suffix}:latest"


def _normalize_instance(instance: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(instance)
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        value = normalized.get(key)
        if isinstance(value, str):
            normalized[key] = json.loads(value)
    required = (
        "instance_id",
        "repo",
        "problem_statement",
        "base_commit",
        "version",
        "test_patch",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
    )
    missing = [key for key in required if normalized.get(key) is None]
    if missing:
        raise ValueError(f"SWE-Gym instance {normalized.get('instance_id', '?')} missing fields: " + ", ".join(missing))
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        if not isinstance(normalized[key], list):
            raise ValueError(f"{key} must be a list for {normalized['instance_id']}")
    return normalized


def load_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyarrow is required to read the SWE-Gym Parquet") from exc

    rows = []
    for row in pq.read_table(path).to_pylist():
        instance = row.get("instance")
        if not isinstance(instance, dict):
            raise ValueError(f"row in {path} does not contain an instance object")
        rows.append(_normalize_instance(instance))
    return rows


def download_split(split: str, output_dir: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise RuntimeError("huggingface_hub is required for --download") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    cached = Path(
        hf_hub_download(
            repo_id=DATASET_NAME,
            filename=f"{split}.parquet",
            repo_type="dataset",
        )
    )
    target = output_dir / f"{split}.parquet"
    if cached.resolve() != target.resolve():
        target.write_bytes(cached.read_bytes())
    return target


def _activate_pinned_package_root(package_root: Path | None) -> None:
    """Expose a pinned checkout without executing swegym's heavy top-level import."""

    if package_root is None:
        return
    package_dir = package_root.resolve() / "swegym"
    if not package_dir.is_dir():
        raise RuntimeError(f"invalid SWE-Gym package root: {package_root}")

    loaded = sys.modules.get("swegym")
    if loaded is not None:
        if str(package_dir) not in getattr(loaded, "__path__", ()):
            raise RuntimeError(f"a different swegym package is already imported; expected {package_dir}")
        return

    # Avoid swegym/__init__.py: it imports optional Docker/collection packages
    # that are irrelevant to this converter.
    package = types.ModuleType("swegym")
    package.__file__ = str(package_dir / "__init__.py")
    package.__package__ = "swegym"
    package.__path__ = [str(package_dir)]
    sys.modules["swegym"] = package


def load_test_spec_builder(
    package_root: Path | None = None,
) -> Callable[[dict[str, Any]], Any]:
    _activate_pinned_package_root(package_root)
    try:
        from swegym.harness.test_spec import make_test_spec
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"The pinned swegym harness is required; install SWE-Bench-Package at commit {SWEGYM_HARNESS_COMMIT} or pass --swegym-package-root") from exc
    return make_test_spec


def load_official_log_parser_source(package_root: Path | None = None) -> str:
    """Load and make self-contained the pinned repo-specific parser map."""

    _activate_pinned_package_root(package_root)
    try:
        from swegym.harness import log_parsers
    except ModuleNotFoundError as exc:
        raise RuntimeError("The pinned swegym log parsers are required; install SWE-Bench-Package or pass --swegym-package-root") from exc
    source_path = Path(str(log_parsers.__file__))
    return standalone_log_parser_source(source_path.read_text(encoding="utf-8"))


def build_eval_command(
    *,
    eval_script: str,
    instance: dict[str, Any],
    timeout: int,
    log_parser_source: str,
) -> str:
    script_b64 = base64.b64encode(zlib.compress(eval_script.encode(), level=9)).decode()
    parser_b64 = base64.b64encode(zlib.compress(log_parser_source.encode(), level=9)).decode()
    expected_b64 = base64.b64encode(
        zlib.compress(
            json.dumps(
                {
                    "FAIL_TO_PASS": instance["FAIL_TO_PASS"],
                    "PASS_TO_PASS": instance["PASS_TO_PASS"],
                },
                separators=(",", ":"),
            ).encode(),
            level=9,
        )
    ).decode()
    instance_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(instance["instance_id"]))
    eval_path = f"/tmp/miles_swegym_eval_{instance_slug}.sh"
    log_path = f"/tmp/miles_swegym_eval_{instance_slug}.log"
    decoder = "import base64,zlib,sys;open(sys.argv[1],'wb').write(zlib.decompress(base64.b64decode(sys.argv[2])))"
    parser = "import base64,zlib;exec(zlib.decompress(base64.b64decode(" + repr(parser_b64) + ")))"
    return (
        f"python3 -c {shlex.quote(decoder)} {shlex.quote(eval_path)} {shlex.quote(script_b64)}; "
        f"chmod 0700 {shlex.quote(eval_path)}; "
        f"timeout {int(timeout)} bash {shlex.quote(eval_path)} >{shlex.quote(log_path)} 2>&1; "
        "RC=$?; "
        f"python3 -c {shlex.quote(parser)} {shlex.quote(log_path)} "
        f'{shlex.quote(expected_b64)} {shlex.quote(str(instance["repo"]))} "$RC"'
    )


def build_git_sanitizer_command(
    *,
    base_commit: str,
    workdir: str,
    timeout: int = DEFAULT_GIT_SANITIZE_TIMEOUT,
) -> dict[str, Any]:
    """Build the before-agent cleanup that removes task-history leakage."""

    target = str(base_commit).strip()
    if not target:
        raise ValueError("base_commit is required for SWE-Gym git sanitization")

    quoted_workdir = shlex.quote(str(workdir).strip() or "/testbed")
    script = "; ".join(
        [
            "set -e",
            f"cd {quoted_workdir}",
            "if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then echo 'SWE-Gym git sanitize failed: not a git worktree' >&2; exit 2; fi",
            f"TARGET={shlex.quote(target)}",
            "git config --global --add safe.directory " + quoted_workdir + " >/dev/null 2>&1 || true",
            'if ! git cat-file -e "$TARGET^{commit}" >/dev/null 2>&1; then echo "SWE-Gym git sanitize failed: missing base commit $TARGET" >&2; exit 2; fi',
            'RESOLVED=$(git rev-parse "$TARGET^{commit}")',
            'git reset --hard "$RESOLVED"',
            # Keep ignored dependencies/caches intact while removing any
            # untracked source or test artifacts baked into the task image.
            "git clean -ffd",
            'git checkout --detach "$RESOLVED"',
            'for remote in $(git remote); do git remote remove "$remote" || true; done',
            # With a detached HEAD, every named ref is unnecessary for solving
            # the task and may retain a future fix, stash, tag, or alternate
            # branch. Delete them uniformly before expiring reflogs.
            "git for-each-ref --format='delete %(refname)' | git update-ref --stdin",
            "git reflog expire --expire=now --expire-unreachable=now --all",
            "git gc --prune=now",
            "git prune --expire=now",
            'test "$(git rev-parse HEAD)" = "$RESOLVED"',
            'test -z "$(git status --porcelain --untracked-files=all)"',
            'echo "SWE-Gym git sanitize completed at $RESOLVED"',
        ]
    )
    return {
        "name": GIT_SANITIZE_CMD_NAME,
        "cmd": "bash -lc " + shlex.quote(script),
        "timeout": int(timeout),
        "required": True,
    }
