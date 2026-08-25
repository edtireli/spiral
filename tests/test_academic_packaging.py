"""Regression coverage for the installed academic fine-tuning commands."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path


ROOT = Path(__file__).parents[1]
ENTRY_POINTS = {
    "spiral-academic-corpus": "scripts.academic_finetune.build_corpus:main",
    "spiral-academic-train": "scripts.academic_finetune.train_qlora:main",
    "spiral-academic-eval": "scripts.academic_finetune.evaluate:main",
    "spiral-academic-serve": "scripts.academic_finetune.serve_adapter:main",
    "spiral-academic-vlm-serve": "scripts.academic_finetune.serve_vlm_adapter:main",
}
REQUIRED_PACKAGE_FILES = {
    "scripts/__init__.py",
    "scripts/academic_finetune/__init__.py",
    "scripts/academic_finetune/adapter_manifest.schema.json",
    "scripts/academic_finetune/build_corpus.py",
    "scripts/academic_finetune/corpus.py",
    "scripts/academic_finetune/evaluate.py",
    "scripts/academic_finetune/fixtures/arxiv_hep_ph.xml",
    "scripts/academic_finetune/fixtures/arxiv_hep_th.xml",
    "scripts/academic_finetune/fixtures/pubmed.xml",
    "scripts/academic_finetune/qwen38_27b_q4.toml",
    "scripts/academic_finetune/requirements-mlx.lock",
    "scripts/academic_finetune/requirements-vlm.lock",
    "scripts/academic_finetune/serve_adapter.py",
    "scripts/academic_finetune/serve_vlm_adapter.py",
    "scripts/academic_finetune/sources.py",
    "scripts/academic_finetune/text.py",
    "scripts/academic_finetune/train_qlora.py",
    "scripts/academic_finetune/training_support.py",
}


def test_academic_package_and_entry_points_survive_wheel_install(tmp_path: Path) -> None:
    """Build without networking, then test the wheel outside the source tree."""

    wheel_dir = tmp_path / "wheel"
    offline_environment = os.environ.copy()
    offline_environment.update({"PIP_NO_INDEX": "1", "UV_OFFLINE": "1"})
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(wheel_dir),
        ],
        cwd=ROOT,
        env=offline_environment,
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1

    install_root = tmp_path / "installed"
    with zipfile.ZipFile(wheels[0]) as archive:
        names = set(archive.namelist())
        assert REQUIRED_PACKAGE_FILES <= names
        assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)
        metadata_paths = [name for name in names if name.endswith(".dist-info/entry_points.txt")]
        assert len(metadata_paths) == 1
        metadata = archive.read(metadata_paths[0]).decode("utf-8")
        for command, target in ENTRY_POINTS.items():
            assert f"{command} = {target}" in metadata
        archive.extractall(install_root)

    check_code = textwrap.dedent(
        """
        import importlib.metadata
        import sys

        install_root = sys.argv[1]
        sys.path.insert(0, install_root)
        distribution = next(
            item
            for item in importlib.metadata.distributions(path=[install_root])
            if item.metadata["Name"] == "spiral-coder"
        )
        expected = {
            "spiral-academic-corpus": "scripts.academic_finetune.build_corpus:main",
            "spiral-academic-train": "scripts.academic_finetune.train_qlora:main",
            "spiral-academic-eval": "scripts.academic_finetune.evaluate:main",
            "spiral-academic-serve": "scripts.academic_finetune.serve_adapter:main",
            "spiral-academic-vlm-serve": "scripts.academic_finetune.serve_vlm_adapter:main",
        }
        actual = {
            entry.name: entry
            for entry in distribution.entry_points
            if entry.group == "console_scripts" and entry.name in expected
        }
        assert {name: entry.value for name, entry in actual.items()} == expected
        for name in sorted(expected):
            assert callable(actual[name].load()), name
        """
    )
    subprocess.run(
        [sys.executable, "-I", "-c", check_code, str(install_root)],
        cwd=tmp_path,
        env=offline_environment,
        check=True,
        capture_output=True,
        text=True,
    )
