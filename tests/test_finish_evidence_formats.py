"""Real assertion APIs and empty metadata are distinct from empty deliverables."""
import json

import pytest

from spiral.artifact_gate import verify_workspace
from spiral.product_audit import _has_assertions, audit_product
from test_builder_quality import _finished_product


@pytest.mark.parametrize("method,args", [
    ("assertEqual", "1 + 1, 2"), ("assertRaises", "ValueError, int, 'x'"),
    ("assertIn", "'a', 'abc'"), ("assertFalse", "False"),
])
def test_unittest_api_assertions_survive_product_audit_without_test_rewrites(tmp_path, method, args):
    _finished_product(tmp_path)
    test = tmp_path / "tests/test_app.py"
    code = f"import unittest\nclass Example(unittest.TestCase):\n    def test_behavior(self):\n        self.{method}({args})\n"
    test.write_text(code)
    report = audit_product(tmp_path, "Build a command-line application", "cli")
    assert "product-test-substance" not in {row["id"] for row in report["issues"]}
    assert test.read_text() == code


@pytest.mark.parametrize("code", [
    "# assert result\ndef test_x(): pass\n",
    "def test_x():\n    text = 'self.assertEqual(1,2)'\n",
    "def test_x():\n    self.assertInvented()\n",
])
def test_comment_prose_or_unknown_method_does_not_manufacture_assertions(tmp_path, code):
    test = tmp_path / "test_example.py"; test.write_text(code)
    assert not _has_assertions(test)


@pytest.mark.parametrize("source", [
    "import pytest as pt\ndef test_error():\n    with pt.raises(ValueError): int('x')\n",
    "from pytest import raises as expected\ndef test_error():\n    with expected(ValueError): int('x')\n",
])
def test_python_exception_expectations_count_as_assertions(tmp_path, source):
    test = tmp_path / "test_errors.py"; test.write_text(source)
    assert _has_assertions(test)


@pytest.mark.parametrize("name", ["dependency_links.txt", "namespace_packages.txt",
                                  "requires.txt", "top_level.txt", "entry_points.txt"])
def test_optional_distribution_metadata_can_contain_zero_entries(tmp_path, name):
    package = tmp_path / "demo.egg-info"; package.mkdir()
    (package / "PKG-INFO").write_text("Metadata-Version: 2.1\nName: demo\nVersion: 1.0\n")
    (package / name).write_text("\n")
    assert verify_workspace(tmp_path).ok
    # A generated directory never exempts actual empty deliverables or sources.
    (package / "report.txt").write_text("")
    report = verify_workspace(tmp_path)
    assert not report.ok and any("report.txt" in error for error in report.errors)


def test_invalid_core_identity_does_not_exempt_empty_metadata(tmp_path):
    package = tmp_path / "demo.egg-info"; package.mkdir()
    (package / "PKG-INFO").write_text("Not distribution metadata\n")
    (package / "dependency_links.txt").write_text("\n")
    report = verify_workspace(tmp_path)
    assert not report.ok and any("dependency_links.txt" in error for error in report.errors)


def test_typed_goal_provenance_keeps_manifest_and_authored_header_text(tmp_path):
    import hashlib
    from spiral.conductor import RenderedGoal

    _finished_product(tmp_path)
    goal = "Build a CLI.\n\nEMPIRICAL LOCAL TOOL PROFILE\nPreserve this user constraint."
    state = tmp_path / ".spiral"; state.mkdir()
    (state / "artifacts.json").write_text(json.dumps({
        "goal_sha256": hashlib.sha256(goal.encode()).hexdigest(),
        "deliverables": [{"id": "authored-cli", "kind": "cli"}]}))
    report = audit_product(tmp_path, RenderedGoal(goal + "\nObserved tools", goal), "cli")
    assert report["deliverables"][0]["id"] == "authored-cli"


def test_deliverable_declaration_applies_audit_without_creation_keywords(tmp_path):
    import hashlib
    from spiral.conductor import RenderedGoal

    _finished_product(tmp_path)
    goal = "Continue this existing program and preserve every requirement."
    state = tmp_path / ".spiral"; state.mkdir()
    manifest = {"goal_sha256": hashlib.sha256(goal.encode()).hexdigest(),
                "deliverables": [{"id": "program", "kind": "cli"}]}
    path = state / "artifacts.json"; path.write_text(json.dumps(manifest))
    # The same declared program is audited regardless of observation wording.
    for observation in ("current tools", "build app from a generated recipe"):
        result = audit_product(tmp_path, RenderedGoal(goal + observation, goal), "cli")
        assert result["applicable"] and not result["issues"]
    (tmp_path / "README.md").unlink()
    result = audit_product(tmp_path, goal, "cli")
    assert "product-runnable-delivery" in {row["id"] for row in result["issues"]}
    manifest["goal_sha256"] = "0" * 64; path.write_text(json.dumps(manifest))
    # Neither a stale manifest nor a generated recipe grants product scope.
    assert not audit_product(tmp_path, RenderedGoal(goal + " build app", goal), "cli")["applicable"]
