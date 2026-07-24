"""Recorded, non-executing acquisition of public reference repositories for Builder."""
from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path


_GITHUB = re.compile(
    r"^https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?$"
)
_LICENSE_NAMES = (
    "LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING", "COPYING.md",
)
_PROJECT_MARKERS = {
    "gradlew": 10, "package.json": 9, "pyproject.toml": 9, "Cargo.toml": 9,
    "go.mod": 9, "pom.xml": 9, "mvnw": 9, "Package.swift": 9,
    "lakefile.lean": 9, "lakefile.toml": 9, "CMakeLists.txt": 9,
    "Makefile": 8, "pytest.ini": 7, "requirements.txt": 2, "index.html": 2,
    "mix.exs": 9, "pubspec.yaml": 9, "Gemfile": 8, "composer.json": 8,
    "build.zig": 9, "meson.build": 9, "MODULE.bazel": 9, "WORKSPACE": 8,
    "DESCRIPTION": 7, "Project.toml": 8, "main.tf": 7,
    "build.gradle": 9, "build.gradle.kts": 9, "settings.gradle": 8,
    "stack.yaml": 9, "cabal.project": 9, "dune-project": 9,
    "build.sbt": 9, "deps.edn": 8, "shard.yml": 8,
}
_PROJECT_IGNORES = {
    ".git", ".spiral", ".venv", "venv", "node_modules", "dist", "build",
    "target", "coverage", "__pycache__", ".pytest_cache",
}
_DYNAMIC_PROJECT_PATTERNS = {
    "*.sln": 9, "*.csproj": 9, "*.fsproj": 9, "*.vbproj": 9,
    "*.xcodeproj": 9, "*.xcworkspace": 9, "*.cabal": 9, "*.nimble": 9,
}


def discover_project_roots(workspace: str | Path, *, max_depth: int = 3) -> list[Path]:
    """Rank plausible runnable project roots, including a generated ``project/``.

    Builders commonly start in an empty directory and place the actual product one
    level down. Verification, dependency installation, audits, and screenshots must
    all agree on that root or they end up inspecting different products.
    """

    root = Path(workspace).resolve()
    candidates: set[Path] = {root}
    for current, dirs, files in os.walk(root):
        here = Path(current)
        try:
            depth = len(here.relative_to(root).parts)
        except ValueError:
            continue
        dirs[:] = [
            name for name in dirs
            if name not in _PROJECT_IGNORES and not name.startswith(".")
            and depth < max_depth
        ]
        if depth > max_depth:
            continue
        if any(name in _PROJECT_MARKERS for name in files):
            candidates.add(here)
        if any(
            any(here.glob(pattern)) for pattern in _DYNAMIC_PROJECT_PATTERNS
        ):
            candidates.add(here)
        if "tests" in dirs or "test" in dirs:
            candidates.add(here)

    def score(path: Path) -> tuple[int, int, str]:
        points = sum(weight for name, weight in _PROJECT_MARKERS.items()
                     if (path / name).is_file())
        points += max(
            [weight for pattern, weight in _DYNAMIC_PROJECT_PATTERNS.items()
             if any(path.glob(pattern))] or [0]
        )
        if (path / "tests").is_dir() or (path / "test").is_dir():
            points += 4
        if any((path / name).is_file() for name in (
                "README.md", "README.rst", "README.txt")):
            points += 1
        depth = len(path.relative_to(root).parts)
        return points, -depth, str(path)

    return sorted((path for path in candidates if score(path)[0] > 0),
                  key=score, reverse=True) or [root]


def primary_project_root(workspace: str | Path) -> Path:
    return discover_project_roots(workspace)[0]


def runnable_project_roots(workspace: str | Path) -> list[Path]:
    """Roots whose dependency manifests participate in the verification graph."""

    roots: list[Path] = []
    weak: list[Path] = []
    strong = {
        "gradlew", "package.json", "pyproject.toml", "Cargo.toml", "go.mod",
        "pom.xml", "mvnw", "Package.swift", "lakefile.lean", "lakefile.toml",
        "CMakeLists.txt", "Makefile", "pytest.ini",
        "mix.exs", "pubspec.yaml", "Gemfile", "composer.json", "build.zig",
        "meson.build", "MODULE.bazel", "WORKSPACE", "DESCRIPTION",
        "Project.toml", "main.tf",
        "build.gradle", "build.gradle.kts", "settings.gradle",
        "stack.yaml", "cabal.project", "dune-project", "build.sbt",
        "deps.edn", "shard.yml",
    }
    for root in discover_project_roots(workspace):
        if any((root / marker).exists() for marker in strong):
            roots.append(root)
            continue
        if any(any(root.glob(pattern)) for pattern in _DYNAMIC_PROJECT_PATTERNS):
            roots.append(root)
            continue
        if any(root.glob("requirements*.txt")) and any(root.rglob("*.py")):
            weak.append(root)
    # A stray requirements file beside a real nested product is often failed
    # scaffolding from an earlier attempt. It must not hijack dependency setup.
    # Weak Python roots are used only when no native project root exists.
    return roots or weak or [primary_project_root(workspace)]


def _size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            pass
    return total


def _license(repo: Path) -> dict:
    path = next((repo / name for name in _LICENSE_NAMES if (repo / name).is_file()), None)
    if path is None:
        return {"status": "unknown", "file": "", "name": "unknown"}
    text = path.read_text(errors="replace")[:40_000]
    low = text.lower()
    if "mit license" in low or "permission is hereby granted, free of charge" in low:
        name, status = "MIT", "permissive"
    elif "apache license" in low and "version 2.0" in low:
        name, status = "Apache-2.0", "permissive"
    elif "redistribution and use in source and binary forms" in low:
        name, status = "BSD", "permissive"
    elif "isc license" in low:
        name, status = "ISC", "permissive"
    elif any(token in low for token in (
            "gnu general public license", "gnu affero general public license",
            "mozilla public license", "eclipse public license")):
        name, status = "copyleft", "restricted-review-only"
    else:
        name, status = "unclassified", "unknown"
    return {"status": status, "file": path.name, "name": name,
            "sha256": __import__("hashlib").sha256(text.encode()).hexdigest()}


def _env(cache: Path) -> dict[str, str]:
    home = cache / "_home"
    home.mkdir(parents=True, exist_ok=True)
    env = {
        key: value for key, value in os.environ.items()
        if key in {"PATH", "LANG", "LC_ALL", "LC_CTYPE", "SSL_CERT_FILE",
                   "SSL_CERT_DIR", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"}
    }
    env.update({
        "HOME": str(home),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "/usr/bin/false",
    })
    return env


def _summary(manifest: dict, repo: Path) -> str:
    readme = next((p for p in (
        repo / "README.md", repo / "README.rst", repo / "README.txt", repo / "README"
    ) if p.is_file()), None)
    excerpt = readme.read_text(errors="replace")[:6000] if readme else "(no README)"
    tree = []
    for item in sorted(repo.rglob("*")):
        if not item.is_file() or ".git" in item.parts:
            continue
        tree.append(str(item.relative_to(repo)))
        if len(tree) >= 120:
            tree.append("...(tree truncated)")
            break
    license_info = manifest.get("license") or {}
    copying = (
        "Source reuse may be considered with attribution and license compliance."
        if license_info.get("status") == "permissive"
        else "Treat as study-only; do not copy or vendor source because the license is not permissive-confirmed."
    )
    return (
        "PUBLIC REPOSITORY ACQUIRED FOR INSPECTION ONLY\n"
        f"URL: {manifest.get('url')}\nCOMMIT: {manifest.get('head')}\n"
        f"CACHE: {repo}\nSIZE: {manifest.get('bytes')} bytes\n"
        f"LICENSE: {license_info.get('name')} ({license_info.get('status')}). {copying}\n"
        "The harness did not execute repository code. Prefer consuming a released package "
        "through the project's normal dependency system over copying implementation.\n\n"
        "TREE:\n" + "\n".join(tree) + "\n\nREADME EXCERPT:\n" + excerpt
    )


def acquire_public_repo(url: str, workspace: str | Path, *, max_mb: int = 500) -> str:
    """Clone one public GitHub repo into ``.spiral/tools`` and return an audit summary.

    Acquisition is credential-free and shallow. The repository is never executed. A
    partial or oversized clone is removed before returning an error.
    """

    url = str(url or "").strip().rstrip("/")
    match = _GITHUB.fullmatch(url)
    if not match:
        return "repo request rejected: use exactly https://github.com/owner/repo"
    owner, name = match.groups()
    name = name.removesuffix(".git")
    cache = Path(workspace).resolve() / ".spiral" / "tools"
    cache.mkdir(parents=True, exist_ok=True)
    repo = cache / f"{owner}-{name}"
    manifest_path = cache / f"{owner}-{name}.json"
    if repo.is_dir() and manifest_path.is_file():
        try:
            return _summary(json.loads(manifest_path.read_text()), repo)
        except Exception:
            shutil.rmtree(repo, ignore_errors=True)
            manifest_path.unlink(missing_ok=True)
    if shutil.disk_usage(cache).free < max_mb * 1024 * 1024:
        return f"repo request rejected: less than {max_mb} MiB free disk"
    git = shutil.which("git")
    if not git:
        return "repo request failed: git is not installed"
    env = _env(cache)
    cmd = [
        git, "-c", "credential.helper=", "-c", "core.hooksPath=/dev/null",
        "clone", "--depth", "1", "--filter=blob:none", url, str(repo),
    ]
    try:
        result = subprocess.run(
            cmd, cwd=cache, capture_output=True, text=True, stdin=subprocess.DEVNULL,
            timeout=600, env=env,
        )
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout).splitlines()[-1])
        size = _size(repo)
        if size > max_mb * 1024 * 1024:
            raise RuntimeError(f"clone exceeds {max_mb} MiB limit")
        head = subprocess.run(
            [git, "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True,
            stdin=subprocess.DEVNULL, timeout=30, env=env,
        )
        if head.returncode != 0:
            raise RuntimeError("could not record repository commit")
        manifest = {
            "schema_version": 1,
            "url": url,
            "head": head.stdout.strip(),
            "bytes": size,
            "license": _license(repo),
            "acquired_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "credential_environment": "scrubbed",
            "execution": "never executed by acquisition harness",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return _summary(manifest, repo)
    except Exception as exc:
        shutil.rmtree(repo, ignore_errors=True)
        manifest_path.unlink(missing_ok=True)
        return f"repo request failed and partial clone was removed: {type(exc).__name__}: {exc}"


def promote_public_repo(
    url: str, workspace: str | Path, *, max_mb: int = 500,
) -> str:
    """Promote a pinned permissive repo from inspection cache to offline execution.

    Promotion does not execute code. It removes git metadata, rejects symlinks and
    records the exact source commit. The command broker is the only supported
    execution path afterward.
    """

    url = str(url or "").strip().rstrip("/")
    match = _GITHUB.fullmatch(url)
    if not match:
        return "repo promotion rejected: use exactly https://github.com/owner/repo"
    owner, name = match.groups()
    name = name.removesuffix(".git")
    root = Path(workspace).resolve()
    cache = root / ".spiral" / "tools"
    repo = cache / f"{owner}-{name}"
    manifest_path = cache / f"{owner}-{name}.json"
    if not repo.is_dir() or not manifest_path.is_file():
        acquired = acquire_public_repo(url, root, max_mb=max_mb)
        if not repo.is_dir() or not manifest_path.is_file():
            return acquired
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
        return f"repo promotion rejected: invalid acquisition manifest: {exc}"
    license_info = manifest.get("license") or {}
    if license_info.get("status") != "permissive":
        return (
            "repo promotion rejected: execution requires a confirmed permissive "
            f"license, found {license_info.get('name', 'unknown')}"
        )
    links = [
        str(path.relative_to(repo))
        for path in repo.rglob("*") if path.is_symlink()
    ]
    if links:
        return (
            "repo promotion rejected: repository contains symlinks that could escape "
            f"the execution tree ({', '.join(links[:5])})"
        )
    commit = str(manifest.get("head") or "")
    destination = (
        root / ".spiral" / "tool-runs"
        / f"{owner}-{name}-{commit[:12] or 'unknown'}"
    )
    if destination.exists():
        return (
            f"PROMOTED_PATH: {destination.relative_to(root)}\n"
            f"pinned commit {commit}; license {license_info.get('name')}; "
            "execute only through ASK: shell with network denied"
        )
    if shutil.disk_usage(root).free < max(64 * 1024 * 1024, int(manifest.get("bytes") or 0) * 2):
        return "repo promotion rejected: insufficient free disk for an isolated execution copy"
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            repo, destination,
            ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"),
        )
        promotion = {
            "schema_version": 1,
            "url": url,
            "source_commit": commit,
            "source_manifest": str(manifest_path.relative_to(root)),
            "license": license_info,
            "promoted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "execution_policy": (
                "credential-scrubbed, outbound network denied, workspace transaction active"
            ),
        }
        (destination / ".spiral-promotion.json").write_text(
            json.dumps(promotion, indent=2), encoding="utf-8")
        return (
            f"PROMOTED_PATH: {destination.relative_to(root)}\n"
            f"pinned commit {commit}; license {license_info.get('name')}; "
            "inspect its README, then execute only the narrow command needed via ASK: shell"
        )
    except Exception as exc:
        shutil.rmtree(destination, ignore_errors=True)
        return (
            "repo promotion failed and the isolated copy was removed: "
            f"{type(exc).__name__}: {exc}"
        )


def ensure_node_dependencies(workspace: str | Path, *, timeout: int = 900,
                             allow_scripts: bool = False) -> dict:
    """Synchronize JS dependencies after a model changes ``package.json``.

    The environment is scrubbed and lifecycle scripts are disabled by default. This
    gives Builder real package/tool access without handing arbitrary postinstall code
    API keys, SSH agents, or the user's home directory.
    """

    root = Path(workspace).resolve()
    package = root / "package.json"
    if not package.is_file():
        return {"applicable": False, "ok": True}
    cache = root / ".spiral" / "dependency-cache"
    cache.mkdir(parents=True, exist_ok=True)
    lockfiles = [
        root / name for name in ("pnpm-lock.yaml", "yarn.lock", "bun.lock", "bun.lockb", "package-lock.json")
        if (root / name).is_file()
    ]
    digest = hashlib.sha256(package.read_bytes())
    for lock in lockfiles:
        digest.update(lock.name.encode())
        digest.update(lock.read_bytes())
    wanted = digest.hexdigest()
    state_path = cache / "state.json"
    try:
        state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    except Exception:
        state = {}
    if ((root / "node_modules").is_dir() and state.get("ok") is True
            and state.get("input_sha256") == wanted):
        return {"applicable": True, "ok": True, "changed": False,
                "detail": "node dependencies already synchronized"}

    try:
        package_data = json.loads(package.read_text())
    except Exception as exc:
        return {"applicable": True, "ok": False,
                "detail": f"package.json is invalid: {exc}"}
    registry_name = re.compile(r"^(?:@[A-Za-z0-9_.-]+/)?[A-Za-z0-9_.-]+$")
    forbidden_spec = re.compile(
        r"(?:https?://|git(?:\+|hub:|lab:)?|ssh:|file:|portal:|patch:)",
        re.I,
    )
    for section in (
        "dependencies", "devDependencies", "optionalDependencies", "peerDependencies",
    ):
        values = package_data.get(section) or {}
        if not isinstance(values, dict):
            return {"applicable": True, "ok": False,
                    "detail": f"package.json {section} must be an object"}
        for dependency, version in values.items():
            if not registry_name.fullmatch(str(dependency)):
                return {"applicable": True, "ok": False,
                        "detail": f"non-registry Node dependency name requires review: {dependency}"}
            spec = str(version).strip()
            if forbidden_spec.search(spec):
                return {"applicable": True, "ok": False,
                        "detail": f"non-registry Node dependency requires manual review: {dependency}@{spec}"}
            if ("/" in spec or "\\" in spec) and not spec.startswith(("workspace:", "npm:")):
                return {"applicable": True, "ok": False,
                        "detail": f"path-based Node dependency requires manual review: {dependency}@{spec}"}
    package_manager = str(package_data.get("packageManager") or "").split("@", 1)[0]

    # A project-level token would be read by the package manager despite the scrubbed
    # process environment. Refuse that boundary rather than accidentally recording or
    # transmitting a credential during an unattended install.
    for name in (".npmrc", ".yarnrc", ".yarnrc.yml"):
        config = root / name
        if config.is_file():
            config_text = config.read_text(errors="replace")
            if re.search(
                    r"(?:_authToken|npmAuthToken|npmAuthIdent|_password)\s*[:=]",
                    config_text, re.I):
                return {"applicable": True, "ok": False,
                        "detail": f"automatic install refused: {name} contains registry credentials"}
            registries = re.findall(
                r"(?:registry|npmRegistryServer)\s*[:=]\s*[\"']?([^\s\"']+)",
                config_text, re.I,
            )
            if any(
                    not value.startswith(("https://registry.npmjs.org", "https://registry.yarnpkg.com"))
                    for value in registries):
                return {"applicable": True, "ok": False,
                        "detail": f"automatic install refused: {name} selects a custom registry"}

    package_lock = root / "package-lock.json"
    if package_lock.is_file():
        try:
            lock_data = json.loads(package_lock.read_text())
            resolved_urls = []

            def collect(value):
                if isinstance(value, dict):
                    if isinstance(value.get("resolved"), str):
                        resolved_urls.append(value["resolved"])
                    for child in value.values():
                        collect(child)
                elif isinstance(value, list):
                    for child in value:
                        collect(child)

            collect(lock_data)
        except Exception as exc:
            return {"applicable": True, "ok": False,
                    "detail": f"package-lock.json is invalid: {exc}"}
        bad_urls = [
            value for value in resolved_urls
            if value.startswith(("http://", "https://"))
            and not value.startswith("https://registry.npmjs.org/")
        ]
        if bad_urls:
            return {"applicable": True, "ok": False,
                    "detail": f"package lock references a nonstandard registry: {bad_urls[0]}"}
    for lock_name in ("pnpm-lock.yaml", "yarn.lock"):
        lock = root / lock_name
        if not lock.is_file():
            continue
        urls = re.findall(r"https?://[^\s\"']+", lock.read_text(errors="replace"))
        bad = [
            value for value in urls
            if not value.startswith((
                "https://registry.npmjs.org/",
                "https://registry.yarnpkg.com/",
            ))
        ]
        if bad:
            return {"applicable": True, "ok": False,
                    "detail": f"{lock_name} references a nonstandard registry: {bad[0]}"}

    def manager(name: str) -> list[str] | None:
        direct = shutil.which(name)
        if direct:
            return [direct]
        corepack = shutil.which("corepack")
        if corepack and name in {"pnpm", "yarn"}:
            return [corepack, name]
        return None

    wanted_manager = (
        "pnpm" if (root / "pnpm-lock.yaml").is_file() or package_manager == "pnpm"
        else "yarn" if (root / "yarn.lock").is_file() or package_manager == "yarn"
        else "bun" if ((root / "bun.lock").is_file() or (root / "bun.lockb").is_file()
                           or package_manager == "bun")
        else "npm"
    )
    runner = manager(wanted_manager)
    if not runner:
        return {"applicable": True, "ok": False,
                "detail": f"package.json requires {wanted_manager}, but it is not installed"}
    if wanted_manager == "pnpm":
        cmd = [*runner, "install", "--no-frozen-lockfile"]
        if not allow_scripts:
            cmd.append("--ignore-scripts")
    elif wanted_manager == "yarn":
        cmd = [*runner, "install"]
        if not allow_scripts:
            cmd.append("--ignore-scripts")
    elif wanted_manager == "bun":
        cmd = [*runner, "install"]
        if not allow_scripts:
            cmd.append("--ignore-scripts")
    else:
        cmd = [*runner, "install", "--no-audit", "--no-fund"]
        if not allow_scripts:
            cmd.append("--ignore-scripts")

    home = cache / "home"
    npm_cache = cache / "npm"
    home.mkdir(exist_ok=True)
    npm_cache.mkdir(exist_ok=True)
    env = {
        key: value for key, value in os.environ.items()
        if key in {"PATH", "LANG", "LC_ALL", "LC_CTYPE", "SSL_CERT_FILE", "SSL_CERT_DIR",
                   "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"}
    }
    env.update({
        "HOME": str(home),
        "npm_config_cache": str(npm_cache),
        "npm_config_userconfig": os.devnull,
        "COREPACK_HOME": str(cache / "corepack"),
        "CI": "1",
    })
    if not allow_scripts:
        env.update({
            "npm_config_ignore_scripts": "true",
            "YARN_ENABLE_SCRIPTS": "false",
            "PNPM_IGNORE_SCRIPTS": "true",
        })
    started = time.time()
    try:
        result = subprocess.run(
            [str(part) for part in cmd], cwd=root, capture_output=True, text=True,
            stdin=subprocess.DEVNULL, timeout=timeout, env=env,
        )
        ok = result.returncode == 0
        # npm may have created/updated the lockfile, so hash the final inputs.
        final_digest = hashlib.sha256(package.read_bytes())
        for name in ("pnpm-lock.yaml", "yarn.lock", "bun.lock", "bun.lockb", "package-lock.json"):
            lock = root / name
            if lock.is_file():
                final_digest.update(lock.name.encode())
                final_digest.update(lock.read_bytes())
        record = {
            "schema_version": 1,
            "input_sha256": final_digest.hexdigest() if ok else wanted,
            "command": [Path(str(cmd[0])).name, *[str(part) for part in cmd[1:]]],
            "lifecycle_scripts": "allowed" if allow_scripts else "disabled",
            "credential_environment": "scrubbed",
            "returncode": result.returncode,
            "seconds": round(time.time() - started, 2),
            "stdout_tail": (result.stdout or "")[-4000:],
            "stderr_tail": (result.stderr or "")[-4000:],
            "ok": ok,
        }
        state_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return {"applicable": True, "ok": ok, "changed": True,
                "detail": ("dependencies synchronized" if ok else
                           (result.stderr or result.stdout or "dependency install failed")[-1200:]),
                "manifest": str(state_path)}
    except Exception as exc:
        return {"applicable": True, "ok": False,
                "detail": f"dependency synchronization failed: {type(exc).__name__}: {exc}"}


def _python_requirements(root: Path) -> tuple[list[str], list[Path], list[str]]:
    """Read ordinary declarative Python dependencies without executing project code."""

    dependencies: list[str] = []
    inputs: list[Path] = []
    errors: list[str] = []
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        inputs.append(pyproject)
        try:
            data = tomllib.loads(pyproject.read_text())
            project = data.get("project") or {}
            dependencies.extend(str(item).strip() for item in project.get("dependencies") or [])
            optional = project.get("optional-dependencies") or {}
            for group in ("dev", "test", "tests", "testing", "quality"):
                dependencies.extend(str(item).strip() for item in optional.get(group) or [])
            groups = data.get("dependency-groups") or {}
            for group in ("dev", "test", "tests", "testing", "quality"):
                dependencies.extend(
                    str(item).strip() for item in groups.get(group) or []
                    if isinstance(item, str)
                )
        except Exception as exc:
            errors.append(f"invalid pyproject.toml: {exc}")

    seen: set[Path] = set()

    def read_requirements(path: Path) -> None:
        path = path.resolve()
        if path in seen:
            return
        seen.add(path)
        try:
            path.relative_to(root)
        except ValueError:
            errors.append(f"requirements include escapes workspace: {path}")
            return
        if not path.is_file():
            errors.append(f"included requirements file is missing: {path.relative_to(root)}")
            return
        inputs.append(path)
        for number, raw in enumerate(path.read_text(errors="replace").splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            line = line.split(" #", 1)[0].strip()
            include = re.match(r"^(?:-r|--requirement)\s+(.+)$", line)
            if include:
                read_requirements((path.parent / include.group(1).strip()).resolve())
                continue
            if line.startswith("-"):
                errors.append(f"unsupported pip option in {path.relative_to(root)}:{number}: {line}")
                continue
            if re.match(r"^(?:<{7}|={7}|>{7})(?:\s|$)", line):
                errors.append(
                    f"unresolved merge-conflict marker in "
                    f"{path.relative_to(root)}:{number}: {line}"
                )
                continue
            dependencies.append(line)

    for path in sorted(root.glob("requirements*.txt")):
        read_requirements(path)

    clean: list[str] = []
    try:
        from packaging.requirements import InvalidRequirement, Requirement
    except Exception:
        try:
            from pip._vendor.packaging.requirements import (  # type: ignore
                InvalidRequirement, Requirement,
            )
        except Exception:
            InvalidRequirement = ValueError
            Requirement = None
    for requirement in dependencies:
        try:
            parsed = Requirement(requirement) if Requirement else None
        except InvalidRequirement:
            parsed = None
        if parsed is None or parsed.url:
            errors.append(f"non-registry Python dependency requires manual review: {requirement}")
            continue
        if requirement not in clean:
            clean.append(requirement)
    return clean, sorted(set(inputs)), errors


def ensure_python_dependencies(workspace: str | Path, *, timeout: int = 900,
                               allow_source_builds: bool = False) -> dict:
    """Install declared Python dependencies into an isolated project-local venv."""

    root = Path(workspace).resolve()
    requirements, inputs, errors = _python_requirements(root)
    if errors:
        return {"applicable": bool(inputs), "ok": False, "detail": "; ".join(errors[:6])}
    if not requirements:
        return {"applicable": False, "ok": True}

    cache = root / ".spiral" / "dependency-cache" / "python"
    venv = cache / "venv"
    state_path = cache / "state.json"
    cache.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    for path in inputs:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    digest.update("\n".join(requirements).encode())
    wanted = digest.hexdigest()
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    bindir = python.parent
    run_env = {
        "PATH": str(bindir) + os.pathsep + os.environ.get("PATH", ""),
        "VIRTUAL_ENV": str(venv),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    try:
        state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    except Exception:
        state = {}
    if python.is_file() and state.get("input_sha256") == wanted:
        return {"applicable": True, "ok": True, "changed": False,
                "detail": "Python dependencies already synchronized", "environment": run_env}

    env = {
        key: value for key, value in os.environ.items()
        if key in {"PATH", "LANG", "LC_ALL", "LC_CTYPE", "SSL_CERT_FILE", "SSL_CERT_DIR",
                   "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"}
    }
    env.update({
        "HOME": str(cache / "home"),
        "PIP_CACHE_DIR": str(cache / "pip-cache"),
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INPUT": "1",
        "PYTHONNOUSERSITE": "1",
    })
    Path(env["HOME"]).mkdir(exist_ok=True)
    Path(env["PIP_CACHE_DIR"]).mkdir(exist_ok=True)
    started = time.time()
    try:
        if not python.is_file():
            made = subprocess.run(
                [sys.executable, "-m", "venv", str(venv)], cwd=root,
                capture_output=True, text=True, stdin=subprocess.DEVNULL,
                timeout=min(timeout, 180), env=env,
            )
            if made.returncode != 0:
                raise RuntimeError(made.stderr or made.stdout or "could not create Python venv")
        cmd = [
            str(python), "-m", "pip", "install", "--disable-pip-version-check",
            "--no-input", "--upgrade",
        ]
        if not allow_source_builds:
            cmd.append("--only-binary=:all:")
        cmd.extend(requirements)
        result = subprocess.run(
            cmd, cwd=root, capture_output=True, text=True, stdin=subprocess.DEVNULL,
            timeout=timeout, env=env,
        )
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout or "pip install failed")[-3000:])
        record = {
            "schema_version": 1,
            "input_sha256": wanted,
            "requirements": requirements,
            "source_builds": "allowed" if allow_source_builds else "binary-wheels-only",
            "credential_environment": "scrubbed",
            "seconds": round(time.time() - started, 2),
            "stdout_tail": (result.stdout or "")[-4000:],
            "stderr_tail": (result.stderr or "")[-4000:],
            "ok": True,
        }
        state_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return {"applicable": True, "ok": True, "changed": True,
                "detail": f"synchronized {len(requirements)} Python requirement(s)",
                "manifest": str(state_path), "environment": run_env}
    except Exception as exc:
        state_path.unlink(missing_ok=True)
        return {"applicable": True, "ok": False,
                "detail": f"Python dependency synchronization failed: {type(exc).__name__}: {exc}"}


def ensure_rust_dependencies(workspace: str | Path, *, timeout: int = 900) -> dict:
    """Fetch crates without running project build scripts or accepting git registries."""

    root = Path(workspace).resolve()
    manifest = root / "Cargo.toml"
    if not manifest.is_file():
        return {"applicable": False, "ok": True}
    cargo = shutil.which("cargo")
    if not cargo:
        return {"applicable": True, "ok": False, "detail": "Cargo.toml exists but cargo is unavailable"}
    try:
        data = tomllib.loads(manifest.read_text())
    except Exception as exc:
        return {"applicable": True, "ok": False, "detail": f"Cargo.toml is invalid: {exc}"}

    errors = []

    def inspect_dependencies(section: dict, label: str) -> None:
        for name, spec in (section or {}).items():
            if not isinstance(spec, dict):
                continue
            if spec.get("git") or spec.get("registry"):
                errors.append(
                    f"{label}.{name} uses a git or alternate-registry dependency")
            if spec.get("path"):
                candidate = (root / str(spec["path"])).resolve()
                try:
                    candidate.relative_to(root)
                except ValueError:
                    errors.append(f"{label}.{name} path dependency escapes the project")
                if not candidate.exists():
                    errors.append(f"{label}.{name} path dependency is missing")

    for key in ("dependencies", "dev-dependencies", "build-dependencies"):
        inspect_dependencies(data.get(key) or {}, key)
    inspect_dependencies((data.get("workspace") or {}).get("dependencies") or {},
                         "workspace.dependencies")
    for target, target_data in (data.get("target") or {}).items():
        for key in ("dependencies", "dev-dependencies", "build-dependencies"):
            inspect_dependencies(
                (target_data or {}).get(key) or {}, f"target.{target}.{key}")

    cargo_config = next((
        path for path in (
            root / ".cargo" / "config.toml", root / ".cargo" / "config",
        ) if path.is_file()
    ), None)
    if cargo_config:
        try:
            config = tomllib.loads(cargo_config.read_text())
            if config.get("registries") or config.get("registry") or config.get("source"):
                errors.append(
                    f"{cargo_config.relative_to(root)} selects a custom registry/source")
        except Exception as exc:
            errors.append(f"invalid {cargo_config.relative_to(root)}: {exc}")
    lock = root / "Cargo.lock"
    if lock.is_file():
        sources = re.findall(r'^source\s*=\s*"([^"]+)"', lock.read_text(), re.M)
        bad = [
            source for source in sources
            if source not in {
                "registry+https://github.com/rust-lang/crates.io-index",
                "sparse+https://index.crates.io/",
            }
        ]
        if bad:
            errors.append(f"Cargo.lock uses a non-crates.io source: {bad[0]}")
    if errors:
        return {"applicable": True, "ok": False, "detail": "; ".join(errors[:8])}

    cache = root / ".spiral" / "dependency-cache" / "rust"
    cache.mkdir(parents=True, exist_ok=True)
    state_path = cache / "state.json"

    def digest() -> str:
        value = hashlib.sha256(manifest.read_bytes())
        if lock.is_file():
            value.update(lock.read_bytes())
        if cargo_config:
            value.update(cargo_config.read_bytes())
        return value.hexdigest()

    wanted = digest()
    try:
        state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    except Exception:
        state = {}
    cargo_home = cache / "cargo-home"
    environment = {
        "CARGO_HOME": str(cargo_home),
        "PATH": os.environ.get("PATH", ""),
    }
    rustup_home = Path.home() / ".rustup"
    if rustup_home.is_dir():
        environment["RUSTUP_HOME"] = str(rustup_home)
    if state.get("ok") is True and state.get("input_sha256") == wanted:
        return {
            "applicable": True, "ok": True, "changed": False,
            "detail": "Rust dependencies already synchronized",
            "environment": environment, "manifest": str(state_path),
        }

    cargo_home.mkdir(parents=True, exist_ok=True)
    env = {
        key: value for key, value in os.environ.items()
        if key in {"PATH", "LANG", "LC_ALL", "LC_CTYPE", "SSL_CERT_FILE", "SSL_CERT_DIR"}
    }
    env.update({
        "HOME": str(cache / "home"),
        "CARGO_HOME": str(cargo_home),
        "CARGO_NET_GIT_FETCH_WITH_CLI": "false",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
        "CARGO_TERM_COLOR": "never",
        **({"RUSTUP_HOME": str(rustup_home)} if rustup_home.is_dir() else {}),
    })
    Path(env["HOME"]).mkdir(exist_ok=True)
    started = time.time()
    try:
        command = [cargo, "fetch"]
        if lock.is_file():
            command.append("--locked")
        result = subprocess.run(
            command, cwd=root, capture_output=True, text=True,
            stdin=subprocess.DEVNULL, timeout=timeout, env=env,
        )
        ok = result.returncode == 0
        record = {
            "schema_version": 1,
            "input_sha256": digest() if ok else wanted,
            "command": [Path(cargo).name, *command[1:]],
            "registry_policy": "crates.io only; git dependencies rejected",
            "project_code_executed": False,
            "credential_environment": "scrubbed",
            "returncode": result.returncode,
            "seconds": round(time.time() - started, 2),
            "stdout_tail": (result.stdout or "")[-3000:],
            "stderr_tail": (result.stderr or "")[-3000:],
            "ok": ok,
        }
        state_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return {
            "applicable": True, "ok": ok, "changed": True,
            "detail": (
                "Rust dependencies synchronized"
                if ok else (result.stderr or result.stdout or "cargo fetch failed")[-1200:]
            ),
            "environment": environment, "manifest": str(state_path),
        }
    except Exception as exc:
        state_path.unlink(missing_ok=True)
        return {
            "applicable": True, "ok": False,
            "detail": f"Rust dependency synchronization failed: {type(exc).__name__}: {exc}",
        }


def ensure_go_dependencies(workspace: str | Path, *, timeout: int = 900) -> dict:
    """Resolve public Go modules through the checksum-verified public proxy only."""

    root = Path(workspace).resolve()
    manifest = root / "go.mod"
    if not manifest.is_file():
        return {"applicable": False, "ok": True}
    go = shutil.which("go")
    if not go:
        return {"applicable": True, "ok": False, "detail": "go.mod exists but go is unavailable"}
    text = manifest.read_text(errors="replace")
    errors = []
    if re.search(r"\b(?:https?|ssh|git)://|git@", text, re.I):
        errors.append("go.mod contains a direct URL/VCS dependency")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("replace ") or "=>" not in stripped:
            continue
        target = stripped.split("=>", 1)[1].strip().split()[0]
        if target.startswith((".", "/")):
            candidate = (root / target).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                errors.append(f"go.mod replace path escapes the project: {target}")
            if not candidate.exists():
                errors.append(f"go.mod replace path is missing: {target}")
    if errors:
        return {"applicable": True, "ok": False, "detail": "; ".join(errors[:8])}

    sum_file = root / "go.sum"
    cache = root / ".spiral" / "dependency-cache" / "go"
    cache.mkdir(parents=True, exist_ok=True)
    state_path = cache / "state.json"

    def digest() -> str:
        value = hashlib.sha256(manifest.read_bytes())
        if sum_file.is_file():
            value.update(sum_file.read_bytes())
        return value.hexdigest()

    wanted = digest()
    try:
        state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    except Exception:
        state = {}
    go_cache = cache / "build-cache"
    module_cache = cache / "module-cache"
    environment = {
        "GOCACHE": str(go_cache),
        "GOMODCACHE": str(module_cache),
        "GOPROXY": "https://proxy.golang.org",
        "GOSUMDB": "sum.golang.org",
        "GOPRIVATE": "",
        "GONOSUMDB": "",
        "PATH": os.environ.get("PATH", ""),
    }
    if state.get("ok") is True and state.get("input_sha256") == wanted:
        return {
            "applicable": True, "ok": True, "changed": False,
            "detail": "Go dependencies already synchronized",
            "environment": environment, "manifest": str(state_path),
        }

    go_cache.mkdir(parents=True, exist_ok=True)
    module_cache.mkdir(parents=True, exist_ok=True)
    env = {
        key: value for key, value in os.environ.items()
        if key in {"PATH", "LANG", "LC_ALL", "LC_CTYPE", "SSL_CERT_FILE", "SSL_CERT_DIR"}
    }
    env.update({
        **environment,
        "HOME": str(cache / "home"),
        "GOENV": "off",
        "GOTOOLCHAIN": "local",
        "GOVCS": "*:off",
    })
    Path(env["HOME"]).mkdir(exist_ok=True)
    started = time.time()
    try:
        command = [go, "mod", "download", "all"]
        result = subprocess.run(
            command, cwd=root, capture_output=True, text=True,
            stdin=subprocess.DEVNULL, timeout=timeout, env=env,
        )
        ok = result.returncode == 0
        record = {
            "schema_version": 1,
            "input_sha256": digest() if ok else wanted,
            "command": [Path(go).name, "mod", "download", "all"],
            "registry_policy": "proxy.golang.org + sum.golang.org; direct VCS disabled",
            "project_code_executed": False,
            "credential_environment": "scrubbed",
            "returncode": result.returncode,
            "seconds": round(time.time() - started, 2),
            "stdout_tail": (result.stdout or "")[-3000:],
            "stderr_tail": (result.stderr or "")[-3000:],
            "ok": ok,
        }
        state_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return {
            "applicable": True, "ok": ok, "changed": True,
            "detail": (
                "Go dependencies synchronized"
                if ok else (result.stderr or result.stdout or "go mod download failed")[-1200:]
            ),
            "environment": environment, "manifest": str(state_path),
        }
    except Exception as exc:
        state_path.unlink(missing_ok=True)
        return {
            "applicable": True, "ok": False,
            "detail": f"Go dependency synchronization failed: {type(exc).__name__}: {exc}",
        }


def ensure_builder_dependencies(workspace: str | Path, *, timeout: int = 900,
                                allow_scripts: bool = False) -> dict:
    """Synchronize supported ecosystems and return the environment for build gates."""

    workspace = Path(workspace).resolve()
    project_roots = runnable_project_roots(workspace)
    reports = []
    for project_root in project_roots:
        rel = str(project_root.relative_to(workspace) or Path("."))
        for ecosystem, report in (
            ("node", ensure_node_dependencies(
                project_root, timeout=timeout, allow_scripts=allow_scripts)),
            ("python", ensure_python_dependencies(
                project_root, timeout=timeout,
                allow_source_builds=allow_scripts)),
            ("rust", ensure_rust_dependencies(
                project_root, timeout=timeout)),
            ("go", ensure_go_dependencies(
                project_root, timeout=timeout)),
        ):
            reports.append({
                **report, "ecosystem": ecosystem,
                "project_root": str(project_root), "relative_root": rel,
            })
    applicable = [report for report in reports if report.get("applicable")]
    failures = [report for report in applicable if not report.get("ok")]
    environment: dict[str, str] = {}
    for report in applicable:
        for key, value in (report.get("environment") or {}).items():
            if key == "PATH" and environment.get("PATH"):
                current = environment["PATH"].split(os.pathsep)
                environment["PATH"] = os.pathsep.join([
                    *[part for part in str(value).split(os.pathsep)
                      if part not in current],
                    *current,
                ])
            elif key != "VIRTUAL_ENV" or "VIRTUAL_ENV" not in environment:
                environment[key] = value
    return {
        "applicable": bool(applicable),
        "ok": not failures,
        "changed": any(report.get("changed") for report in applicable),
        "detail": (
            "; ".join(
                f"{report.get('relative_root')}:{report.get('ecosystem')} "
                f"{report.get('detail', '')}"
                for report in applicable
            )
            if applicable else ""
        ),
        "environment": environment,
        "reports": reports,
        "project_root": str(project_roots[0]),
        "project_roots": [str(root) for root in project_roots],
    }


def ensure_playwright_chromium(workspace: str | Path, *, timeout: int = 900) -> dict:
    """Ensure Spiral's shared, auditable Chromium runtime exists for visual QA."""

    root = Path(workspace).resolve()
    manifest = root / ".spiral" / "dependency-cache" / "playwright.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    browser_cache = Path.home() / ".cache" / "spiral" / "playwright"
    browser_cache.mkdir(parents=True, exist_ok=True)
    environment = {"PLAYWRIGHT_BROWSERS_PATH": str(browser_cache)}
    try:
        old = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(browser_cache)
        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as runtime:
                executable = Path(runtime.chromium.executable_path)
        finally:
            if old is None:
                os.environ.pop("PLAYWRIGHT_BROWSERS_PATH", None)
            else:
                os.environ["PLAYWRIGHT_BROWSERS_PATH"] = old
    except Exception as exc:
        return {"ok": False, "detail": f"Playwright runtime unavailable: {exc}",
                "environment": environment}
    if executable.is_file():
        return {"ok": True, "changed": False, "detail": "Chromium already installed",
                "environment": environment, "executable": str(executable)}
    system_candidates = [
        Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
        Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
        Path("/usr/bin/google-chrome"), Path("/usr/bin/chromium"),
        Path("/usr/bin/chromium-browser"), Path("/usr/bin/microsoft-edge"),
    ]
    system_browser = next((path for path in system_candidates if path.is_file()), None)
    if shutil.disk_usage(browser_cache).free < 1024 * 1024 * 1024:
        return {"ok": False, "detail": "less than 1 GiB free for the Chromium runtime",
                "environment": environment}

    env = {
        key: value for key, value in os.environ.items()
        if key in {"PATH", "LANG", "LC_ALL", "LC_CTYPE", "SSL_CERT_FILE", "SSL_CERT_DIR",
                   "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"}
    }
    env.update(environment)
    started = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            cwd=root, capture_output=True, text=True, stdin=subprocess.DEVNULL,
            timeout=timeout, env=env,
        )
        record = {
            "schema_version": 1,
            "command": [Path(sys.executable).name, "-m", "playwright", "install", "chromium"],
            "credential_environment": "scrubbed",
            "shared_cache": str(browser_cache),
            "returncode": result.returncode,
            "seconds": round(time.time() - started, 2),
            "stdout_tail": (result.stdout or "")[-4000:],
            "stderr_tail": (result.stderr or "")[-4000:],
            "ok": result.returncode == 0,
        }
        manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")
        if result.returncode != 0:
            if system_browser:
                return {
                    "ok": True, "changed": False,
                    "detail": (
                        "managed Chromium install failed; using system browser "
                        f"{system_browser.name}"
                    ),
                    "environment": environment, "executable": str(system_browser),
                    "manifest": str(manifest),
                }
            return {"ok": False,
                    "detail": (result.stderr or result.stdout or "Chromium install failed")[-1200:],
                    "environment": environment, "manifest": str(manifest)}
        return {
            "ok": True, "changed": True, "detail": "Chromium installed for visual QA",
            "environment": environment, "manifest": str(manifest),
            "executable": str(executable) if executable.is_file() else "",
        }
    except Exception as exc:
        if system_browser:
            return {
                "ok": True, "changed": False,
                "detail": (
                    f"managed Chromium install failed ({type(exc).__name__}); "
                    f"using system browser {system_browser.name}"
                ),
                "environment": environment, "executable": str(system_browser),
            }
        return {"ok": False, "detail": f"Chromium install failed: {type(exc).__name__}: {exc}",
                "environment": environment}
