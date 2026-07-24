"""Executable research certificates for ``spiral research``.

Some research claims are too large for a fixed verifier kind. A sigma-model
classification, for example, may need a custom Gröbner script, a case splitter, a
symbolic RG-flow check, and a reproducibility bundle. This module gives the model a
small but general workbench: write files into a certificate directory, run declared
Python, C/C++, Rust, Go, Julia, R, Java, Swift, Lean, Sage, or Singular commands,
optionally install Python requirements into a local target, and
record hashes/stdout/stderr in a manifest.

The certificate code is still untrusted model output. On macOS it executes under a
kernel sandbox with reads limited to its certificate plus runtime/toolchain roots, no
network access, and writes confined to its certificate directory. The runner also
rejects absolute/parent output paths, avoids shell execution, and screens obviously
destructive/network commands. Other platforms record that no OS sandbox is available
rather than pretending lexical screening is a security boundary. A passing program does
not make a claim true by itself; it makes the claim auditable and re-runnable.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class WorkbenchResult:
    ok: bool
    detail: str
    manifest: str = ""
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    extra: dict = field(default_factory=dict)


_BAD_COMMAND = (
    "rm", "mv", "chmod", "chown", "sudo", "curl", "wget", "ssh", "scp",
    "rsync", "osascript", "open", "kill", "pkill", "launchctl", "mail",
    "mailx", "sendmail", "nc", "ncat", "socat", "gh", "git",
    "bash", "zsh", "sh", "fish",
)
_BAD_TEXT = (
    "shutil.rmtree", "os.remove", "os.unlink", "os.system", "subprocess.",
    "import subprocess", "from subprocess import",
    "socket.", "urllib.request", "requests.", "httpx.", "eval(", "exec(",
    "open('/", 'open("/', "../", "/etc/", "/var/", "/Users/",
    "#include<sys/socket.h>", "#include <sys/socket.h>", "#include<curl/", "#include <curl/",
    "std::filesystem::remove", "std::remove(", "remove_all(", "system(",
    "downloads.download", "http.get", "http.post", "socket.connect", "run(`rm",
    "file.delete", "unlink(", "download.file", "url(", "java.net.",
)
_PY_REQ = re.compile(r"^[A-Za-z0-9_.-]+([<>=!~]=?[A-Za-z0-9_.+*:-]+)?$")
_ALLOWED_PYPI = {
    "allensdk", "astropy", "bids-validator", "biopython", "brainspace",
    "casadi", "cvxpy", "cython", "dask", "einsteinpy", "gmpy2", "h5py",
    "iminuit", "jax", "matplotlib", "mpmath", "netcdf4", "networkx",
    "neuromaps", "nibabel", "nilearn", "numba", "numpy", "nwbinspector",
    "openpyxl", "pandas", "pillow", "pingouin", "polars", "pulp", "pyarrow",
    "pybids", "pydantic", "pynrrd", "pynwb", "pytest", "qutip", "rsatoolbox",
    "scikit-image", "scikit-learn", "scipy", "seaborn", "shapely",
    "statsmodels", "sympy", "templateflow", "torch", "torchvision", "trimesh",
    "xarray", "z3-solver", "zarr",
}
_GITHUB_RE = re.compile(r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?/?$")
_CHECK_SIGNALS = (
    "assert ", "assert(", "raise assertionerror", "pytest", "unittest",
    "simplify(", "groebner(", "allclose(", "isclose(", "residual", "remainder",
    "if (", "if(", "return 1", "return false", "by simp", "by decide", "by ring",
    "@assert", "stopifnot(", "assert!(", "panic!(", "fatalf(", "throw new assertionerror",
)
_SUMMARY_SENSITIVE = re.compile(
    r"(?:subject|participant|patient|person|email|address|phone|birth|"
    r"session_id|scan_id|row_data|raw_data)",
    re.I,
)


def _validation_evidence(validation: dict, step_records: list[dict]) -> dict:
    """Authenticate declared independent methods against distinct executed steps."""

    validation = validation if isinstance(validation, dict) else {}

    def check_items(values, *, prefix: str) -> list[dict]:
        checked = []
        for item in values or []:
            if not isinstance(item, dict):
                checked.append({"declared": item, "valid": False,
                                "reason": "must be an object with name, step, and marker"})
                continue
            name = str(item.get("name") or "").strip()
            marker = str(item.get("marker") or "").strip()
            step = item.get("step")
            if not name or not marker or not isinstance(step, int):
                checked.append({"declared": item, "valid": False,
                                "reason": "missing name/marker/integer step"})
                continue
            if not marker.startswith(prefix):
                checked.append({"declared": item, "valid": False,
                                "reason": f"marker must start with {prefix}"})
                continue
            if step < 0 or step >= len(step_records):
                checked.append({"declared": item, "valid": False,
                                "reason": "step index was not executed"})
                continue
            record = step_records[step]
            lines = set((record.get("stdout_full") or "").splitlines())
            valid = record.get("returncode") == 0 and marker in lines
            checked.append({
                "name": name,
                "step": step,
                "marker": marker,
                "argv": record.get("argv") or [],
                "valid": valid,
                "reason": "" if valid else "successful step did not emit the exact marker",
            })
        return checked

    methods = check_items(
        validation.get("independent_methods") or [], prefix="METHOD_OK:")
    criteria = check_items(
        validation.get("acceptance_criteria") or [], prefix="CRITERION_OK:")
    passed_methods = [item for item in methods if item.get("valid")]
    method_steps = {item.get("step") for item in passed_methods}
    method_commands = {
        tuple(str(part) for part in (item.get("argv") or [])) for item in passed_methods
    }
    independent = (
        len(passed_methods) >= 2
        and len(method_steps) >= 2
        and len(method_commands) >= 2
    )
    criteria_passed = bool(criteria) and all(item.get("valid") for item in criteria)
    return {
        "methods": methods,
        "criteria": criteria,
        "passed_method_count": len(passed_methods),
        "distinct_successful_steps": len(method_steps),
        "distinct_successful_commands": len(method_commands),
        "all_acceptance_criteria_passed": criteria_passed,
        "computationally_reproduced": bool(independent and criteria_passed),
    }


def _result_summary(work: Path) -> dict:
    """Load a bounded aggregate result artifact without exposing row-level data."""

    path = next((
        candidate for candidate in (
            work / "spiral-result.json", work / "result.json")
        if candidate.is_file()
    ), None)
    if path is None:
        return {
            "present": False, "safe": False,
            "reason": "data certificates must write spiral-result.json",
        }
    if path.stat().st_size > 128 * 1024:
        return {
            "present": True, "safe": False,
            "path": path.name, "reason": "aggregate result exceeds 128 KiB",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "present": True, "safe": False, "path": path.name,
            "reason": f"invalid JSON: {type(exc).__name__}: {exc}",
        }
    if not isinstance(payload, dict):
        return {
            "present": True, "safe": False, "path": path.name,
            "reason": "aggregate result must be a JSON object",
        }

    def audit(value, key: str = "", depth: int = 0) -> tuple[bool, str]:
        if depth > 6:
            return False, "aggregate nesting exceeds six levels"
        if key and _SUMMARY_SENSITIVE.search(key):
            return False, f"row-level or identifying field is forbidden: {key}"
        if isinstance(value, dict):
            if len(value) > 100:
                return False, "aggregate object has more than 100 fields"
            for child_key, child in value.items():
                ok, reason = audit(child, str(child_key), depth + 1)
                if not ok:
                    return ok, reason
        elif isinstance(value, list):
            if len(value) > 100:
                return False, "aggregate list has more than 100 values"
            for child in value:
                ok, reason = audit(child, key, depth + 1)
                if not ok:
                    return ok, reason
        elif isinstance(value, str):
            if len(value) > 2000:
                return False, f"aggregate string is too long: {key or 'value'}"
            if value.startswith(("/Users/", "/Volumes/", "/home/")):
                return False, "aggregate contains a host path"
        elif value is not None and not isinstance(value, (bool, int, float)):
            return False, f"unsupported aggregate value: {type(value).__name__}"
        return True, ""

    safe, reason = audit(payload)
    required = {"estimand", "estimate", "uncertainty", "sample_size", "diagnostics"}
    missing = sorted(required - set(payload))
    if missing:
        safe = False
        reason = "missing aggregate fields: " + ", ".join(missing)
    return {
        "present": True, "safe": safe, "path": path.name,
        "reason": reason, "summary": payload if safe else {},
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _dependency_lock(deps: Path) -> list[dict]:
    rows = []
    try:
        for dist in importlib.metadata.distributions(path=[str(deps)]):
            name = str(dist.metadata.get("Name") or "").strip()
            version = str(dist.version or "").strip()
            if name:
                rows.append({"name": name, "version": version})
    except Exception:
        return []
    return sorted(rows, key=lambda row: (row["name"].lower(), row["version"]))


def _tool_identity(executable: str, *, allow_version: bool = True) -> dict:
    path = Path(executable)
    record = {"path": str(path)}
    try:
        record["sha256"] = _sha256(path) if path.is_file() else ""
    except Exception:
        record["sha256"] = ""
    try:
        if not allow_version:
            raise RuntimeError("generated executable version invocation disabled")
        result = subprocess.run(
            [str(path), "--version"], capture_output=True, text=True,
            stdin=subprocess.DEVNULL, timeout=5,
        )
        lines = (result.stdout or result.stderr).splitlines()
        record["version"] = lines[0][:300] if lines else ""
    except Exception:
        record["version"] = ""
    return record


def _artifact_inventory(work: Path) -> list[dict]:
    excluded = {
        "_deps", "_repos", "_data", "_home", "_tmp", "_cache",
        "_acquisition_home",
    }
    rows = []
    for path in work.rglob("*"):
        try:
            rel = path.relative_to(work)
            if not path.is_file() or any(part in excluded for part in rel.parts):
                continue
            if path.name == "manifest.json" or path.stat().st_size > 50 * 1024 * 1024:
                continue
            rows.append({
                "path": str(rel), "sha256": _sha256(path), "bytes": path.stat().st_size,
            })
        except OSError:
            continue
    return sorted(rows, key=lambda row: row["path"])


def _safe_rel(path: str) -> Path:
    p = Path(str(path))
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise ValueError(f"unsafe certificate path: {path!r}")
    return p


def _screen_text(text: str) -> str | None:
    compact = text.replace(" ", "")
    for bad in _BAD_TEXT:
        if bad.replace(" ", "") in compact:
            return bad
    return None


def _static_evidence(files: dict) -> dict:
    """Reject a vacuous script whose only operation is printing a success marker.

    This does not prove the program correct.  It establishes the much smaller but
    important fact that the executable certificate contains a falsifiable check.
    """

    texts = [str(v) for v in files.values()]
    combined = "\n".join(texts).lower()
    lines = [
        line.strip() for text in texts for line in text.splitlines()
        if line.strip() and not line.strip().startswith(("#", "//"))
    ]
    signals = sorted({s for s in _CHECK_SIGNALS if s in combined})
    lean_sorry = any(name.lower().endswith(".lean") and "sorry" in str(content).lower()
                     for name, content in files.items())
    non_marker_lines = [line for line in lines if "certificate_ok" not in line.lower()]
    non_marker_text = re.sub(r"certificate_ok", "", combined, flags=re.I)
    constant_assert = bool(re.search(r"\bassert\s*(?:\(|\s)\s*(?:true|1)\s*\)?\s*[;\n]", combined))
    return {
        "signals": signals,
        "code_lines": len(lines),
        "non_marker_lines": len(non_marker_lines),
        "lean_sorry": lean_sorry,
        "constant_assert": constant_assert,
        "passes": bool(signals and len(non_marker_text.strip()) >= 10
                       and not lean_sorry and not constant_assert),
    }


def _command(
    argv_or_str, work: Path, *, search_path: str | None = None,
) -> list[str]:
    if isinstance(argv_or_str, list):
        argv = [str(x) for x in argv_or_str]
    else:
        argv = shlex.split(str(argv_or_str or ""))
    if not argv:
        raise ValueError("workbench claim needs a command")
    exe_path = Path(argv[0])
    exe = exe_path.name
    if exe in _BAD_COMMAND:
        raise ValueError(f"blocked command: {exe}")
    built_in = {
        "python", "python3", Path(sys.executable).name,
        "pytest", "sage", "lean", "lake",
        "Singular", "singular",
        "cc", "c++", "clang", "clang++", "gcc", "g++",
        "rustc", "go", "julia", "Rscript", "javac", "java",
        "swift", "swiftc",
    }
    if str(argv[0]).startswith("./"):
        rel = _safe_rel(str(argv[0])[2:])
        target = work / rel
        if not target.is_file():
            raise FileNotFoundError(f"local executable is missing: {argv[0]}")
        argv[0] = str(target)
        return argv
    if not re.fullmatch(r"[A-Za-z0-9_.+-]{1,100}", exe):
        raise ValueError(f"unsafe executable name: {exe!r}")
    found = shutil.which(argv[0], path=search_path)
    if exe in {"python", "python3", Path(sys.executable).name}:
        argv[0] = sys.executable
    elif exe in {"lean", "lake"}:
        from spiral.verify_math import _lean_exe

        lean = _lean_exe()
        direct = Path(lean).with_name(exe) if lean else None
        if not direct or not direct.is_file():
            raise FileNotFoundError(f"installed Lean toolchain has no {exe} binary")
        argv[0] = str(direct)
    elif exe.lower() == "singular":
        candidates = [
            Path(found) if found else None,
            Path("/opt/homebrew/opt/singular/bin/Singular"),
            Path("/usr/local/opt/singular/bin/Singular"),
        ]
        direct = next((path for path in candidates if path and path.is_file()), None)
        if direct is None:
            raise FileNotFoundError("Singular is not installed")
        argv[0] = str(direct)
    elif not found:
        raise FileNotFoundError(f"{exe} is not installed")
    else:
        if exe not in built_in and not (
            sys.platform == "darwin" and shutil.which("sandbox-exec")
        ):
            raise ValueError(
                f"general executable {exe!r} requires an OS sandbox; use an "
                "installed built-in scientific toolchain on this host"
            )
        argv[0] = found
    return argv


def _commands(claim: dict) -> list:
    steps = claim.get("steps")
    if steps is not None:
        if not isinstance(steps, list) or not steps:
            raise ValueError("workbench steps must be a non-empty list")
        return steps
    cmd = claim.get("cmd") or claim.get("command")
    if not cmd:
        raise ValueError("workbench claim needs a command")
    return [cmd]


def _execution_command(argv: list[str], work: Path) -> tuple[list[str], dict]:
    """Wrap generated code in the strongest locally available offline sandbox."""

    sandbox = shutil.which("sandbox-exec") if sys.platform == "darwin" else None
    if not sandbox:
        return argv, {
            "mode": "none",
            "network": "not OS-enforced",
            "writes": "not OS-confined",
            "warning": "lexical command screening is defense in depth, not a sandbox",
        }
    def quoted(path: str | Path) -> str:
        return str(Path(path).resolve()).replace("\\", "\\\\").replace('"', '\\"')

    tool = Path(argv[0]).resolve()
    read_roots = {
        str(work.resolve()),
        str(Path(sys.prefix).resolve()),
        str(Path(sys.base_prefix).resolve()),
        str(tool.parent.parent if len(tool.parents) > 1 else tool.parent),
        "/System", "/usr", "/Library", "/opt/homebrew", "/dev",
        "/private/etc", "/private/var/db", "/private/var/select",
        "/System/Volumes/Preboot/Cryptexes",
    }
    read_filters = " ".join(
        f'(subpath "{quoted(path)}")' for path in sorted(read_roots)
        if Path(path).exists()
    )
    metadata_ancestors: set[str] = set()
    for root in read_roots:
        path = Path(root)
        if str(path).startswith(("/Users/", "/Volumes/", "/Network/")):
            for parent in path.parents:
                metadata_ancestors.add(str(parent))
                if str(parent) in {"/Users", "/Volumes", "/Network", "/"}:
                    break
    metadata_filters = " ".join(
        f'(literal "{quoted(path)}")' for path in sorted(metadata_ancestors)
        if Path(path).exists()
    )
    metadata_rule = (
        f'(allow file-read-metadata {metadata_filters}) ' if metadata_filters else "")
    resolved = quoted(work)
    profile = (
        '(version 1) '
        '(deny default) '
        '(allow process*) '
        # Python and compiler runtimes touch changing OS cache locations, so begin
        # with normal system readability, remove user/volume data, then add back only
        # the exact certificate/runtime roots needed by this command.
        '(allow file-read*) '
        '(deny file-read* (subpath "/Users") (subpath "/Volumes") (subpath "/Network")) '
        f'{metadata_rule}'
        f'(allow file-read* {read_filters}) '
        f'(allow file-write* (subpath "{resolved}") (literal "/dev/null")) '
        '(allow sysctl-read)'
    )
    return [sandbox, "-p", profile, *argv], {
        "mode": "macos-sandbox-exec",
        "network": "denied",
        "file_reads": "system-readable; user-and-volume-data-denied except runtime roots",
        "denied_read_roots": ["/Users", "/Volumes", "/Network"],
        "read_roots": sorted(read_roots),
        "runtime_path_metadata_ancestors": sorted(metadata_ancestors),
        "file_writes": "certificate-workdir-only",
        "profile_sha256": hashlib.sha256(profile.encode("utf-8")).hexdigest(),
    }


def _safe_repo_url(url: str) -> str:
    url = str(url or "").strip()
    if not _GITHUB_RE.match(url):
        raise ValueError(f"unsupported repo URL: {url!r}; use public https://github.com/owner/repo")
    return url[:-1] if url.endswith("/") else url


def _repo_name(url: str) -> str:
    name = url.rstrip("/").split("/")[-1]
    if name.endswith(".git"):
        name = name[:-4]
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name)[:80] or "repo"


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def _acquisition_env(work: Path) -> dict[str, str]:
    """A network-capable but credential-free environment for public acquisition."""

    home = work / "_acquisition_home"
    home.mkdir(exist_ok=True)
    env = {
        key: value for key, value in os.environ.items()
        if key in {
            "PATH", "LANG", "LC_ALL", "LC_CTYPE", "SSL_CERT_FILE",
            "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
            "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
        }
    }
    env.update({
        "HOME": str(home),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "/usr/bin/false",
    })
    return env


def _clone_repos(claim: dict, work: Path, *, allow_repos: bool,
                 repo_budget: int, repo_max_mb: int, timeout: float) -> tuple[list[dict], str]:
    repos = claim.get("repos") or []
    if not repos:
        return [], ""
    if not allow_repos:
        raise PermissionError("repo acquisition requires --auto-repos or config research_auto_repos=true")
    if not isinstance(repos, list):
        raise ValueError("repos must be a list")
    if len(repos) > repo_budget:
        raise ValueError(f"repo budget exceeded: {len(repos)} requested, budget {repo_budget}")
    git = shutil.which("git")
    if not git:
        raise FileNotFoundError("git is not installed")

    root = work / "_repos"
    root.mkdir(exist_ok=True)
    env = _acquisition_env(work)
    records = []
    logs = []
    free = shutil.disk_usage(work).free
    if free < max(1, repo_max_mb) * 1024 * 1024:
        raise RuntimeError("not enough free disk for repo acquisition")
    for item in repos:
        if isinstance(item, str):
            item = {"url": item}
        if not isinstance(item, dict):
            raise ValueError("each repo must be a URL string or object")
        url = _safe_repo_url(item.get("url", ""))
        dest = root / _repo_name(url)
        cmd = [git, "clone", "--depth", "1", "--filter=blob:none", url, str(dest)]
        p = subprocess.run(cmd, cwd=work, capture_output=True, text=True,
                           stdin=subprocess.DEVNULL, timeout=min(timeout, 600.0),
                           env=env)
        logs.append((p.stdout + p.stderr)[-3000:])
        if p.returncode != 0:
            shutil.rmtree(dest, ignore_errors=True)
            raise RuntimeError(f"git clone failed for {url}: {(p.stderr or p.stdout).splitlines()[-1:]}")
        ref = str(item.get("ref") or "").strip()
        if ref:
            p = subprocess.run([git, "fetch", "--depth", "1", "origin", ref],
                               cwd=dest, capture_output=True, text=True,
                               stdin=subprocess.DEVNULL, timeout=min(timeout, 300.0),
                               env=env)
            logs.append((p.stdout + p.stderr)[-2000:])
            if p.returncode != 0:
                shutil.rmtree(dest, ignore_errors=True)
                raise RuntimeError(f"git fetch failed for {url}@{ref}")
            p = subprocess.run([git, "checkout", "--detach", "FETCH_HEAD"],
                               cwd=dest, capture_output=True, text=True,
                               stdin=subprocess.DEVNULL, timeout=120, env=env)
            logs.append((p.stdout + p.stderr)[-2000:])
            if p.returncode != 0:
                shutil.rmtree(dest, ignore_errors=True)
                raise RuntimeError(f"git checkout failed for {url}@{ref}")
        size = _dir_size(dest)
        if size > repo_max_mb * 1024 * 1024:
            shutil.rmtree(dest, ignore_errors=True)
            raise RuntimeError(f"repo exceeded size limit: {url} ({size // (1024*1024)} MiB)")
        head = ""
        p = subprocess.run([git, "rev-parse", "HEAD"], cwd=dest,
                           capture_output=True, text=True, stdin=subprocess.DEVNULL,
                           timeout=30, env=env)
        if p.returncode == 0:
            head = p.stdout.strip()
        records.append({
            "url": url, "path": str(dest.relative_to(work)), "head": head,
            "bytes": size, "credential_environment": "scrubbed",
        })
    return records, "\n".join(logs)[-8000:]


def _install_requirements(reqs: list[str], root: Path, timeout: float, *,
                          env: dict[str, str]) -> tuple[bool, str, str, Path]:
    deps = root / "_deps"
    deps.mkdir(exist_ok=True)
    safe = []
    for req in reqs or []:
        r = str(req).strip()
        if not r:
            continue
        if not _PY_REQ.match(r):
            return False, "", f"unsafe requirement: {r}", deps
        package = re.split(r"[<>=!~]", r, 1)[0].lower().replace("_", "-")
        if package not in _ALLOWED_PYPI:
            return False, "", (
                f"requirement is outside the approved research package set: {package}"), deps
        safe.append(r)
    if not safe:
        return True, "", "", deps
    try:
        p = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
             "--no-input", "--no-compile", "--only-binary=:all:",
             "--target", str(deps), *safe],
            cwd=root, capture_output=True, text=True, stdin=subprocess.DEVNULL,
            timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired:
        return False, "", f"pip install timed out after {timeout:g}s", deps
    return p.returncode == 0, p.stdout.strip(), p.stderr.strip(), deps


def run_workbench_claim(claim: dict, root: str | Path, *, timeout: float = 300.0,
                        allow_repos: bool = False, repo_budget: int = 1,
                        repo_max_mb: int = 750,
                        cleanup_failed_repos: bool = True,
                        allow_tools: bool = False, tool_budget: int = 4,
                        allow_data: bool = False, data_root: str | Path | None = None,
                        data_cfg=None) -> WorkbenchResult:
    """Run a model-authored certificate claim in a local research workbench.

    Claim schema:

    ``{"kind":"workbench", "files":{"check.py":"..."}, "cmd":"python check.py",
    "expect":"CERTIFICATE_OK", "requirements":["sympy"], "note":"..."}``

    ``steps`` may replace ``cmd`` for compile/run bundles, e.g. C++:
    ``"steps":["c++ -std=c++17 check.cpp -o check","./check"]``.

    Public GitHub repositories are not cloned by default. With ``allow_repos=True``,
    ``repos`` may list public ``https://github.com/owner/repo`` URLs; they are cloned
    into ``_repos`` inside the certificate workdir, recorded in the manifest, and
    removed again if the certificate fails.

    ``datasets`` may declare typed public data requests. When ``allow_data`` is true,
    the scientific-data broker locks ``analysis_plan`` and ``alignment`` first,
    resolves exact files/bytes/licences, acquires them into a shared immutable cache,
    and hard-links them beneath ``_data/ALIAS`` for offline analysis.
    """
    base = Path(root)
    started = time.monotonic()

    def finish(result: WorkbenchResult) -> WorkbenchResult:
        try:
            from spiral.toolsmith import Toolsmith

            Toolsmith(base).observe_workbench(
                claim, result, max(0.0, time.monotonic() - started))
        except Exception:
            pass
        return result

    slug = re.sub(r"[^a-z0-9]+", "-", str(claim.get("note") or "certificate").lower()).strip("-")[:60]
    stamp = f"{int(time.time() * 1000)}"
    work = base / (slug or "certificate") / stamp
    work.mkdir(parents=True, exist_ok=True)
    manifest = {
        "claim": claim,
        "workdir": str(work),
        "files": [],
        "command": claim.get("cmd") or claim.get("command"),
        "steps": claim.get("steps"),
        "requirements": claim.get("requirements") or [],
        "tools": claim.get("tools") or [],
        "repos": [],
        "datasets": claim.get("datasets") or [],
        "started_at": stamp,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "python_executable": sys.executable,
        },
    }

    try:
        files = claim.get("files") or {}
        if not isinstance(files, dict) or not files:
            raise ValueError("workbench claim needs a non-empty files object")
        for name, content in files.items():
            rel = _safe_rel(name)
            text = str(content)
            bad = _screen_text(text)
            if bad:
                raise ValueError(f"blocked text in {name}: {bad}")
            path = work / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            manifest["files"].append({
                "path": str(rel),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            })
        static_evidence = _static_evidence(files)
        manifest["static_evidence"] = static_evidence
        exploratory = str(claim.get("evidence_mode") or "") == "exploratory"
        manifest["evidence_mode"] = "exploratory" if exploratory else "certificate"
        if not static_evidence["passes"] and not exploratory:
            if static_evidence["lean_sorry"]:
                raise ValueError("certificate contains Lean `sorry`")
            raise ValueError(
                "certificate is vacuous: include a falsifiable assertion/residual/check "
                "before printing CERTIFICATE_OK")

        repo_records, repo_log = _clone_repos(
            claim, work, allow_repos=allow_repos, repo_budget=repo_budget,
            repo_max_mb=repo_max_mb, timeout=timeout)
        manifest["repos"] = repo_records
        manifest["repo_log"] = repo_log[-4000:]

        requested_data = claim.get("datasets") or []
        if requested_data:
            if not allow_data:
                raise PermissionError(
                    "scientific-data acquisition is disabled; enable research_data_auto")
            from spiral.research_data import ScientificDataBroker

            broker_root = Path(data_root).resolve() if data_root else base.parent / "data"
            data_broker = ScientificDataBroker(broker_root, cfg=data_cfg)
            data_evidence = data_broker.acquire_many(
                requested_data,
                plan=claim.get("analysis_plan") or {},
                alignment=claim.get("alignment") or {},
                materialize_to=work / "_data",
            )
            manifest["data_evidence"] = data_evidence
        else:
            manifest["data_evidence"] = {
                "ok": True, "not_applicable": True,
                "confirmatory_ready": True, "provenance_complete": True,
            }

        requested_tools = claim.get("tools") or []
        if not isinstance(requested_tools, list):
            raise ValueError("tools must be a list of typed requests")
        if requested_tools and not allow_tools:
            raise PermissionError(
                "tool provisioning is disabled; enable research_tool_auto")
        if len(requested_tools) > max(0, tool_budget):
            raise ValueError(
                f"tool budget exceeded: {len(requested_tools)} requested, "
                f"budget {tool_budget}")
        from spiral.command_broker import CommandBroker

        broker = CommandBroker(base)
        provisioned = []
        for request in requested_tools:
            if isinstance(request, dict):
                request = (
                    f"{request.get('ecosystem', '')} "
                    f"{request.get('package', '')}"
                ).strip()
            result = broker.provision(str(request), timeout=min(900, int(timeout)))
            provisioned.append({"request": str(request), "result": result})
            if not result.startswith("tool installed:"):
                raise RuntimeError(result)
        manifest["tool_provisioning"] = provisioned

        deps = work / "_deps"
        runtime_home = work / "_home"
        runtime_tmp = work / "_tmp"
        runtime_cache = work / "_cache"
        for path in (deps, runtime_home, runtime_tmp, runtime_cache):
            path.mkdir(exist_ok=True)
        # Do not expose API keys, SSH agents, cloud credentials, or the user's HOME to
        # acquisition/runtime subprocesses.  Only ordinary executable/locale/TLS settings
        # survive.  Dependencies are binary wheels only, so package build scripts never run.
        env = {
            key: value for key, value in os.environ.items()
            if key in {
                "PATH", "LANG", "LC_ALL", "LC_CTYPE", "SSL_CERT_FILE",
                "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
                "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
            }
        }
        env.update({
            "HOME": str(runtime_home),
            "TMPDIR": str(runtime_tmp),
            "XDG_CACHE_HOME": str(runtime_cache),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(deps),
        })
        if broker.environment.get("PATH"):
            env["PATH"] = broker.environment["PATH"]
        install_ok, install_out, install_err, deps = _install_requirements(
            claim.get("requirements") or [], work, min(timeout, 300.0), env=env)
        manifest["install_stdout"] = install_out[-4000:]
        manifest["install_stderr"] = install_err[-4000:]
        manifest["dependency_acquisition"] = {
            "top_level_allowlist": sorted(_ALLOWED_PYPI),
            "binary_wheels_only": True,
            "subprocess_environment": sorted(env),
        }
        if not install_ok:
            raise RuntimeError(install_err or "requirement installation failed")
        manifest["dependency_lock"] = _dependency_lock(deps)

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        step_records: list[dict] = []
        returncode = 0
        for raw_cmd in _commands(claim):
            bad_cmd = _screen_text(str(raw_cmd))
            if bad_cmd:
                raise ValueError(f"blocked command text: {bad_cmd}")
            argv = _command(raw_cmd, work, search_path=env.get("PATH"))
            bad_cmd = _screen_text(" ".join(argv[1:]))
            if bad_cmd:
                raise ValueError(f"blocked command text: {bad_cmd}")
            execution_argv, isolation = _execution_command(argv, work)
            manifest.setdefault("execution_isolation", isolation)
            try:
                p = subprocess.run(execution_argv, cwd=work, capture_output=True,
                                   text=True, stdin=subprocess.DEVNULL,
                                   timeout=timeout, env=env)
            except subprocess.TimeoutExpired as exc:
                manifest["timed_out"] = True
                manifest["stdout"] = (exc.stdout or "")[-8000:] if isinstance(exc.stdout, str) else "\n".join(stdout_parts)[-8000:]
                manifest["stderr"] = (exc.stderr or "")[-8000:] if isinstance(exc.stderr, str) else "\n".join(stderr_parts)[-8000:]
                for record in step_records:
                    record.pop("stdout_full", None)
                manifest["steps_run"] = step_records
                if cleanup_failed_repos and manifest.get("repos"):
                    shutil.rmtree(work / "_repos", ignore_errors=True)
                    manifest["repos_cleaned_after_failure"] = True
                mf = work / "manifest.json"
                mf.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                return finish(WorkbenchResult(
                    False, f"certificate timed out after {timeout:g}s",
                    str(mf), manifest["stdout"], manifest["stderr"], True))
            so, se = (p.stdout or "").strip(), (p.stderr or "").strip()
            stdout_parts.append(so)
            stderr_parts.append(se)
            step_records.append({
                "argv": argv,
                "tool": _tool_identity(
                    argv[0],
                    allow_version=not Path(argv[0]).resolve().is_relative_to(work.resolve())),
                "sandboxed_argv": execution_argv[:3] + ["<command>"]
                if isolation.get("mode") == "macos-sandbox-exec" else execution_argv,
                "returncode": p.returncode,
                "stdout_full": so,
                "stdout_tail": so[-2000:],
                "stderr_tail": se[-2000:],
            })
            returncode = p.returncode
            if p.returncode != 0:
                break
        out, err = "\n".join(x for x in stdout_parts if x).strip(), "\n".join(x for x in stderr_parts if x).strip()
        expect = (
            str(claim.get("expect") or "")
            if "expect" in claim else "CERTIFICATE_OK"
        )
        last = out.splitlines()[-1].strip() if out else ""
        expect_ok = expect == "" or last == expect or expect in out
        marker_ok = "CERTIFICATE_OK" in out.splitlines()
        validation_evidence = _validation_evidence(
            claim.get("validation") or {}, step_records)
        aggregate = _result_summary(work) if requested_data else {
            "present": False, "safe": True, "not_applicable": True,
        }
        if requested_data:
            manifest["data_evidence"]["result_summary"] = aggregate
            manifest["data_evidence"]["result_summary_ready"] = bool(
                aggregate.get("safe"))
        for record in step_records:
            record.pop("stdout_full", None)
        # The contract for generated research certificates is the marker: the code
        # should print CERTIFICATE_OK only after its own independent checks pass. Some
        # local models overfill `expect` with an entire stdout transcript; keep that
        # mismatch auditable, but do not reject an otherwise marked certificate.
        ok = returncode == 0 and (expect_ok or ("CERTIFICATE_OK" in expect and marker_ok))
        if not ok and cleanup_failed_repos and manifest.get("repos"):
            shutil.rmtree(work / "_repos", ignore_errors=True)
            manifest["repos_cleaned_after_failure"] = True
        manifest.update({
            "argv": step_records[-1]["argv"] if step_records else [],
            "steps_run": step_records,
            "returncode": returncode,
            "stdout": out[-12000:],
            "stderr": err[-12000:],
            "expect": expect,
            "expect_matched": expect_ok,
            "marker_matched": marker_ok,
            "validation_evidence": validation_evidence,
            "ok": ok,
            "artifacts": _artifact_inventory(work),
        })
        mf = work / "manifest.json"
        mf.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        detail = f"certificate {'passed' if ok else 'failed'}: {Path(mf).parent}"
        if not ok:
            if returncode != 0:
                detail += f"; returncode {returncode}"
            elif not marker_ok and "CERTIFICATE_OK" in expect:
                detail += "; CERTIFICATE_OK marker was not printed"
            elif not expect_ok:
                detail += "; expected output did not match"
            if err:
                detail += f"; stderr: {err.splitlines()[-1][:160]}"
        return finish(WorkbenchResult(
            ok, detail, str(mf), out, err,
            extra={
                "returncode": returncode, "expect": expect,
                "steps": len(step_records),
                "result_summary": aggregate.get("summary") or {},
                "result_summary_ready": bool(aggregate.get("safe")),
            }))
    except Exception as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        if cleanup_failed_repos and manifest.get("repos"):
            shutil.rmtree(work / "_repos", ignore_errors=True)
            manifest["repos_cleaned_after_failure"] = True
        mf = work / "manifest.json"
        mf.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return finish(WorkbenchResult(False, manifest["error"], str(mf)))
