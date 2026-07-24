"""Cross-domain structural evidence gate for projects without a native test gate."""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import tarfile
import tomllib
import wave
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath


_SKIP = {
    ".git", ".spiral", ".venv", "venv", "node_modules", "dist", "build",
    "target", "coverage", "__pycache__", ".pytest_cache",
}
_TEXT = {
    ".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".json", ".toml",
    ".yaml", ".yml", ".html", ".htm", ".css", ".xml", ".svg", ".md", ".txt",
    ".sh", ".bash", ".zsh", ".lean", ".tex", ".csv", ".sql", ".rs", ".go",
    ".c", ".h", ".cpp", ".hpp", ".java", ".kt", ".swift", ".rb", ".php",
    ".scala", ".dart", ".lua", ".r", ".R", ".jl", ".sol", ".ex", ".exs",
    ".ipynb", ".gltf", ".obj", ".tsv", ".jsonl", ".rst", ".rtf",
}
_IMAGES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}
_MEDIA = {
    ".mp4", ".mov", ".mkv", ".webm", ".mp3", ".wav", ".flac", ".m4a",
    ".ogg", ".aac",
}
_CONFLICT = ("<<<<<<< ", "=======", ">>>>>>> ")
_CONFLICT_TEXT = _TEXT - {".md", ".txt", ".tex", ".csv", ".tsv"}
_ZIP_ARTIFACTS = {
    ".docx": "word/document.xml",
    ".pptx": "ppt/presentation.xml",
    ".xlsx": "xl/workbook.xml",
    ".odt": "content.xml",
    ".odp": "content.xml",
    ".ods": "content.xml",
    ".epub": "META-INF/container.xml",
}
_PACKAGE_ARTIFACTS = {
    ".apk", ".aab", ".ipa", ".jar", ".war", ".whl", ".zip",
    ".nupkg", ".vsix",
}
_TAR_ARTIFACTS = {".tar", ".tgz", ".crate"}
_EXECUTABLE_BINARY = {
    ".exe", ".dll", ".so", ".dylib", ".bin", ".appimage",
}
_KNOWN_BINARY = (
    _IMAGES | _MEDIA | set(_ZIP_ARTIFACTS) | _PACKAGE_ARTIFACTS | _TAR_ARTIFACTS
    | _EXECUTABLE_BINARY
    | {".pdf", ".sqlite", ".sqlite3", ".db", ".npy", ".npz", ".parquet",
       ".glb", ".wasm", ".dmg", ".pkg", ".deb", ".rpm", ".arrow",
       ".blend", ".fbx", ".stl", ".ply", ".msi"}
)


class _HTMLProbe(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = 0

    def handle_starttag(self, tag, attrs):
        self.tags += 1


@dataclass
class ArtifactReport:
    ok: bool
    verified: int
    skipped: int
    errors: list[str]
    evidence: list[str]


def _files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in _SKIP or part.startswith(".") for part in rel.parts):
            continue
        yield path, rel


def _looks_like_text(path: Path) -> bool:
    try:
        if path.stat().st_size > 2_000_000:
            return False
        sample = path.read_bytes()[:8192]
        return b"\x00" not in sample and bool(sample.decode("utf-8").strip())
    except (OSError, UnicodeDecodeError):
        return False


def _executable_format(path: Path) -> str:
    """Recognize ordinary native executables without attempting to run them."""

    try:
        with path.open("rb") as handle:
            header = handle.read(4096)
    except OSError:
        return ""
    if header.startswith(b"\x7fELF") and len(header) >= 20:
        return "ELF"
    if header[:4] in {
        b"\xfe\xed\xfa\xce", b"\xce\xfa\xed\xfe",
        b"\xfe\xed\xfa\xcf", b"\xcf\xfa\xed\xfe",
        b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca",
        b"\xca\xfe\xba\xbf", b"\xbf\xba\xfe\xca",
    }:
        return "Mach-O"
    if header.startswith(b"MZ") and len(header) >= 64:
        offset = int.from_bytes(header[60:64], "little")
        if offset + 4 <= len(header) and header[offset:offset + 4] == b"PE\0\0":
            return "PE"
    if header.startswith(b"!<arch>\n"):
        return "archive"
    return ""


def _safe_archive_name(name: str) -> bool:
    normalized = str(name or "").replace("\\", "/")
    path = PurePosixPath(normalized)
    return bool(
        normalized
        and not normalized.startswith("/")
        and not re.match(r"^[A-Za-z]:/", normalized)
        and ".." not in path.parts
    )


def _validate_zip(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = archive.infolist()
    if not members:
        raise ValueError("package archive is empty")
    if len(members) > 50_000:
        raise ValueError(f"archive has an excessive member count ({len(members)})")
    total = sum(max(0, member.file_size) for member in members)
    compressed = sum(max(1, member.compress_size) for member in members)
    if total > 4 * 1024 * 1024 * 1024:
        raise ValueError("archive expands beyond the 4 GiB validation ceiling")
    if total > 256 * 1024 * 1024 and total / compressed > 500:
        raise ValueError("archive has a suspicious compression expansion ratio")
    for member in members:
        if not _safe_archive_name(member.filename):
            raise ValueError(f"archive contains escaping member {member.filename}")
        mode = (member.external_attr >> 16) & 0o170000
        if mode == stat.S_IFLNK:
            target = archive.read(member).decode("utf-8", errors="replace")
            joined = PurePosixPath(member.filename).parent / target.replace("\\", "/")
            if (
                target.startswith("/")
                or re.match(r"^[A-Za-z]:[/\\]", target)
                or ".." in joined.parts
            ):
                raise ValueError(
                    f"archive symlink escapes its package: {member.filename} -> {target}")
    bad = archive.testzip()
    if bad:
        raise ValueError(f"corrupt archive member: {bad}")
    return members


def verify_workspace(workspace: str | Path) -> ArtifactReport:
    root = Path(workspace).resolve()
    errors: list[str] = []
    evidence: list[str] = []
    verified = skipped = 0
    node = shutil.which("node")
    ffprobe = shutil.which("ffprobe")

    for path, rel in _files(root):
        suffix = path.suffix
        low = (
            ".tgz" if path.name.lower().endswith(".tar.gz")
            else suffix.lower()
        )
        try:
            if path.is_symlink():
                try:
                    target = path.resolve(strict=False)
                    target.relative_to(root)
                except (OSError, ValueError):
                    raise ValueError(
                        "symbolic link resolves outside the workspace")
                verified += 1
                evidence.append(
                    f"{rel}: internal symbolic link resolves inside workspace")
                continue
            executable_format = (
                _executable_format(path)
                if not suffix or low in _EXECUTABLE_BINARY else ""
            )
            if low == ".blend":
                with path.open("rb") as handle:
                    header = handle.read(7)
                if header != b"BLENDER":
                    raise ValueError("invalid Blender file header")
                verified += 1
                evidence.append(f"{rel}: Blender container header validated")
            elif low == ".fbx":
                with path.open("rb") as handle:
                    header = handle.read(256)
                if not (
                    header.startswith(b"Kaydara FBX Binary")
                    or b"FBXHeaderExtension" in header
                ):
                    raise ValueError("invalid FBX header")
                verified += 1
                evidence.append(f"{rel}: FBX container header validated")
            elif low == ".stl":
                size = path.stat().st_size
                with path.open("rb") as handle:
                    header = handle.read(84)
                binary_ok = (
                    len(header) == 84
                    and size == 84 + 50 * int.from_bytes(header[80:84], "little")
                )
                with path.open("rb") as handle:
                    text_sample = handle.read(1_000_000)
                text_ok = (
                    header.lstrip().lower().startswith(b"solid")
                    and b"facet" in text_sample.lower()
                )
                if not (binary_ok or text_ok):
                    raise ValueError("invalid STL triangle structure")
                verified += 1
                evidence.append(
                    f"{rel}: {'binary' if binary_ok else 'ASCII'} STL decoded")
            elif low == ".ply":
                with path.open("rb") as handle:
                    header = handle.read(16_384)
                end = header.find(b"end_header")
                if (
                    not header.startswith(b"ply\n") or end < 0
                    or b"format " not in header[:end]
                    or b"element vertex " not in header[:end]
                ):
                    raise ValueError("invalid PLY header")
                verified += 1
                evidence.append(f"{rel}: PLY geometry header validated")
            elif executable_format:
                if path.stat().st_size < 16:
                    raise ValueError("native executable is truncated")
                verified += 1
                evidence.append(
                    f"{rel}: {executable_format} binary container validated")
            elif (low in _TEXT or not suffix
                    or (low not in _KNOWN_BINARY and _looks_like_text(path))):
                if path.stat().st_size > 2_000_000:
                    skipped += 1
                    continue
                raw = path.read_bytes()
                if b"\x00" in raw:
                    skipped += 1
                    continue
                text = raw.decode("utf-8", errors="strict")
                if low in _CONFLICT_TEXT or path.name.startswith("requirements"):
                    for marker in _CONFLICT:
                        if any(line.startswith(marker) for line in text.splitlines()):
                            raise ValueError(
                                f"unresolved merge-conflict marker {marker.strip()}")
                if low == ".py":
                    compile(text, str(rel), "exec")
                elif low in {".json", ".gltf"} or path.name.endswith(".ipynb"):
                    data = json.loads(text)
                    if path.name.endswith(".ipynb") and not isinstance(data.get("cells"), list):
                        raise ValueError("notebook has no cells array")
                elif low == ".jsonl":
                    rows = [
                        json.loads(line) for line in text.splitlines()
                        if line.strip()
                    ]
                    if not rows:
                        raise ValueError("JSONL contains no records")
                elif low == ".toml":
                    tomllib.loads(text)
                elif low in {".xml", ".svg"}:
                    ET.fromstring(text)
                elif low in {".html", ".htm"}:
                    parser = _HTMLProbe()
                    parser.feed(text)
                    if parser.tags == 0:
                        raise ValueError("HTML contains no elements")
                elif low in {".csv", ".tsv"}:
                    rows = list(csv.reader(
                        text.splitlines(), delimiter="\t" if low == ".tsv" else ","))
                    if not rows or not any(cell.strip() for row in rows for cell in row):
                        raise ValueError("CSV has no data")
                elif low in {".sh", ".bash", ".zsh"}:
                    shell = "/bin/zsh" if low == ".zsh" else "/bin/bash"
                    checked = subprocess.run(
                        [shell, "-n", str(path)], capture_output=True, text=True,
                        stdin=subprocess.DEVNULL, timeout=15,
                    )
                    if checked.returncode:
                        raise ValueError(checked.stderr.strip() or "shell syntax error")
                elif low in {".js", ".mjs", ".cjs"} and node:
                    checked = subprocess.run(
                        [node, "--check", str(path)], capture_output=True, text=True,
                        stdin=subprocess.DEVNULL, timeout=15,
                    )
                    if checked.returncode:
                        raise ValueError(checked.stderr.strip() or "JavaScript syntax error")
                elif not text.strip():
                    raise ValueError("artifact is empty")
                verified += 1
                evidence.append(f"{rel}: parse/integrity check passed")
            elif low in _IMAGES:
                from PIL import Image, ImageStat

                with Image.open(path) as image:
                    image.verify()
                with Image.open(path).convert("RGB") as image:
                    if image.width < 2 or image.height < 2:
                        raise ValueError("image dimensions are empty")
                    stat = ImageStat.Stat(image.resize((64, 64)))
                    if max(hi - lo for lo, hi in stat.extrema) <= 1:
                        raise ValueError("image is effectively pixel-uniform")
                    evidence.append(f"{rel}: {image.width}x{image.height} image decoded")
                verified += 1
            elif low == ".pdf":
                from pypdf import PdfReader

                pages = len(PdfReader(str(path)).pages)
                if pages < 1:
                    raise ValueError("PDF has no pages")
                verified += 1
                evidence.append(f"{rel}: PDF decoded with {pages} page(s)")
            elif low in _ZIP_ARTIFACTS:
                with zipfile.ZipFile(path) as archive:
                    _validate_zip(archive)
                    required = _ZIP_ARTIFACTS[low]
                    if required not in archive.namelist():
                        raise ValueError(f"missing required package member {required}")
                verified += 1
                evidence.append(f"{rel}: packaged {low[1:]} artifact decoded")
            elif low in _PACKAGE_ARTIFACTS:
                with zipfile.ZipFile(path) as archive:
                    members = _validate_zip(archive)
                    names = [member.filename for member in members]
                    if low == ".apk" and "AndroidManifest.xml" not in names:
                        raise ValueError("APK has no AndroidManifest.xml")
                    if low == ".aab" and not any(
                            name.endswith("/manifest/AndroidManifest.xml")
                            for name in names):
                        raise ValueError("AAB has no module AndroidManifest.xml")
                    if low == ".ipa" and not any(
                            name.startswith("Payload/") and name.endswith(".app/Info.plist")
                            for name in names):
                        raise ValueError("IPA has no application Info.plist")
                    if low == ".jar" and not any(
                            name.endswith(".class") for name in names):
                        raise ValueError("JAR contains no class files")
                    if low == ".whl" and not any(
                            ".dist-info/" in name and name.endswith("METADATA")
                            for name in names):
                        raise ValueError("wheel has no dist-info METADATA")
                verified += 1
                evidence.append(
                    f"{rel}: {low[1:]} package decoded ({len(names)} members)")
            elif low in _TAR_ARTIFACTS:
                with tarfile.open(path, "r:*") as archive:
                    members = archive.getmembers()
                    if not members:
                        raise ValueError("archive is empty")
                    if len(members) > 50_000:
                        raise ValueError(
                            f"archive has an excessive member count ({len(members)})")
                    unsafe = [
                        member.name for member in members
                        if not _safe_archive_name(member.name)
                    ]
                    if unsafe:
                        raise ValueError(
                            f"archive contains escaping member {unsafe[0]}")
                    unsafe_links = [
                        f"{member.name} -> {member.linkname}"
                        for member in members if member.issym() or member.islnk()
                        if (
                            not _safe_archive_name(member.linkname)
                            or ".." in (
                                PurePosixPath(member.name).parent
                                / member.linkname.replace("\\", "/")
                            ).parts
                        )
                    ]
                    if unsafe_links:
                        raise ValueError(
                            f"archive contains escaping link {unsafe_links[0]}")
                    expanded = sum(
                        max(0, member.size) for member in members if member.isfile())
                    if expanded > 4 * 1024 * 1024 * 1024:
                        raise ValueError(
                            "archive expands beyond the 4 GiB validation ceiling")
                verified += 1
                evidence.append(
                    f"{rel}: {low[1:]} archive decoded ({len(members)} members)")
            elif low in {".dmg", ".pkg", ".deb", ".rpm"}:
                with path.open("rb") as handle:
                    header = handle.read(16)
                    if low == ".dmg":
                        handle.seek(max(0, path.stat().st_size - 512))
                        trailer = handle.read(4)
                    else:
                        trailer = b""
                valid = (
                    (low == ".dmg" and trailer == b"koly")
                    or (low == ".pkg" and header.startswith(b"xar!"))
                    or (low == ".deb" and header.startswith(b"!<arch>\n"))
                    or (low == ".rpm" and header.startswith(b"\xed\xab\xee\xdb"))
                )
                if not valid:
                    raise ValueError(f"invalid {low[1:].upper()} container header")
                verified += 1
                evidence.append(f"{rel}: {low[1:].upper()} container header validated")
            elif low == ".msi":
                with path.open("rb") as handle:
                    header = handle.read(8)
                if header != b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
                    raise ValueError("invalid MSI compound-file header")
                verified += 1
                evidence.append(f"{rel}: MSI compound-file container validated")
            elif low in {".sqlite", ".sqlite3", ".db"}:
                with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
                    value = connection.execute("PRAGMA integrity_check").fetchone()[0]
                if value != "ok":
                    raise ValueError(f"SQLite integrity check: {value}")
                verified += 1
                evidence.append(f"{rel}: SQLite integrity check passed")
            elif low in {".npy", ".npz"}:
                import numpy as np

                data = np.load(path, allow_pickle=False)
                if low == ".npz":
                    if not data.files:
                        raise ValueError("NumPy archive contains no arrays")
                    data.close()
                elif getattr(data, "size", 0) == 0:
                    raise ValueError("NumPy array is empty")
                verified += 1
                evidence.append(f"{rel}: NumPy data decoded")
            elif low == ".parquet":
                import pyarrow.parquet as pq

                metadata = pq.read_metadata(path)
                if metadata.num_columns < 1:
                    raise ValueError("Parquet file has no columns")
                verified += 1
                evidence.append(
                    f"{rel}: Parquet decoded ({metadata.num_rows} rows, "
                    f"{metadata.num_columns} columns)")
            elif low == ".arrow":
                import pyarrow.ipc as ipc

                try:
                    with path.open("rb") as handle:
                        table = ipc.open_file(handle).read_all()
                except Exception:
                    with path.open("rb") as handle:
                        table = ipc.open_stream(handle).read_all()
                if table.num_columns < 1:
                    raise ValueError("Arrow data has no columns")
                verified += 1
                evidence.append(
                    f"{rel}: Arrow IPC decoded ({table.num_rows} rows, "
                    f"{table.num_columns} columns)")
            elif low == ".glb":
                header = path.read_bytes()[:12]
                if len(header) != 12 or header[:4] != b"glTF":
                    raise ValueError("invalid GLB header")
                length = int.from_bytes(header[8:12], "little")
                if length != path.stat().st_size:
                    raise ValueError("GLB declared length does not match file size")
                verified += 1
                evidence.append(f"{rel}: GLB container header validated")
            elif low == ".wasm":
                if path.read_bytes()[:8] != b"\x00asm\x01\x00\x00\x00":
                    raise ValueError("invalid WebAssembly header")
                verified += 1
                evidence.append(f"{rel}: WebAssembly module header validated")
            elif low == ".wav" and not ffprobe:
                with wave.open(str(path), "rb") as stream:
                    if stream.getnframes() < 1 or stream.getframerate() < 1:
                        raise ValueError("WAV contains no audio frames")
                    seconds = stream.getnframes() / stream.getframerate()
                verified += 1
                evidence.append(f"{rel}: WAV decoded, duration {seconds:.3f}s")
            elif low in _MEDIA and ffprobe:
                checked = subprocess.run(
                    [ffprobe, "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=nw=1:nk=1", str(path)],
                    capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=30,
                )
                if checked.returncode or not checked.stdout.strip():
                    raise ValueError(checked.stderr.strip() or "media probe failed")
                verified += 1
                evidence.append(f"{rel}: media decoded, duration {checked.stdout.strip()}s")
            elif low in _MEDIA:
                raise RuntimeError(
                    "ffprobe is required to decode this media artifact")
            else:
                skipped += 1
        except Exception as exc:
            errors.append(f"{rel}: {type(exc).__name__}: {exc}")

    if verified == 0:
        errors.append("no structurally verifiable artifacts were found")
    return ArtifactReport(not errors and verified > 0, verified, skipped, errors, evidence)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", nargs="?", default=".")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = verify_workspace(args.workspace)
    if args.json:
        print(json.dumps(asdict(report), indent=2))
    else:
        for row in report.evidence[:30]:
            print(f"PASS {row}")
        for row in report.errors:
            print(f"FAIL {row}")
        print(
            f"artifact gate: {report.verified} verified, {report.skipped} skipped, "
            f"{len(report.errors)} errors"
        )
    return 0 if report.ok else (5 if report.verified == 0 else 1)


if __name__ == "__main__":
    raise SystemExit(main())
