import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from spiral.research_data import (
    CatalogRecord,
    DataFile,
    ScientificDataBroker,
    validate_analysis_contract,
)
from spiral.research_loop import ResearchLoop


def _plan(**updates):
    value = {
        "mode": "confirmatory",
        "hypothesis": "The prespecified regional association differs from zero.",
        "primary_outcome": "Spearman rho across atlas parcels",
        "unit_of_analysis": "atlas parcel",
        "inclusion_exclusion": "all parcels with finite values in both maps",
        "confounds": "parcel area and cortical hierarchy",
        "missing_data": "complete-case parcels, count reported",
        "multiple_testing": "one primary test; FDR for labelled secondary tests",
        "spatial_null": "10,000 spatial rotations preserving hemispheres",
        "validation": "repeat in a held-out atlas",
        "replication": "independent held-out atlas and permutation implementation",
        "stopping_rule": "fixed samples and permutations; no optional stopping",
        "causal_scope": "spatial association only",
    }
    value.update(updates)
    return value


def _alignment(**updates):
    value = {
        "target_space": "MNI152",
        "registration": "declared transforms plus landmark and overlap QC",
        "resolution_policy": "resample once to the coarser map",
        "species_bridge": "N/A; both resources are human",
        "participant_linkage": "unmatched group-level sources",
    }
    value.update(updates)
    return value


def test_analysis_contract_requires_spatial_null_and_species_bridge():
    requests = [
        {"source": "openneuro", "id": "ds000001", "species": "human"},
        {"source": "allen", "id": "product-2", "species": "mouse"},
    ]
    report = validate_analysis_contract(
        _plan(spatial_null=""), _alignment(species_bridge=""), requests)
    assert not report["confirmatory_ready"]
    assert "spatial_null" in report["missing_plan_fields"]
    assert "species_bridge" in report["missing_alignment_fields"]


def test_analysis_contract_accepts_prespecified_human_multimodal_study():
    requests = [
        {"source": "openneuro", "id": "ds000001", "species": "human"},
        {"source": "allen", "id": "product-2", "species": "human"},
    ]
    report = validate_analysis_contract(_plan(), _alignment(), requests)
    assert report["passes"]
    assert report["confirmatory_ready"]
    assert report["warnings"]


def test_openneuro_catalog_normalizes_version_doi_and_modality(tmp_path, monkeypatch):
    broker = ScientificDataBroker(tmp_path)
    monkeypatch.setattr(broker, "_graphql", lambda query, variables=None: {
        "advancedSearch": {"edges": [{"node": {
            "id": "ds006072",
            "name": "Psilocybin mapping",
            "public": True,
            "publishDate": "2025-01-01",
            "metadata": {
                "modalities": ["MRI"], "species": "human",
                "datasetUrl": "", "associatedPaperDOI": "10.example/paper",
                "studyDesign": "within subject", "studyDomain": "psychedelics",
                "tasksCompleted": ["rest"],
            },
            "latestSnapshot": {
                "tag": "1.1.0",
                "description": {
                    "Name": "Psilocybin Precision Functional Mapping",
                    "DatasetDOI": "doi:10.18112/openneuro.ds006072.v1.1.0",
                    "License": "CC0", "Authors": ["Researcher"],
                },
            },
        }}]},
    })
    rows = broker._discover_openneuro("psilocybin fMRI receptor", 5)
    assert len(rows) == 1
    assert isinstance(rows[0], CatalogRecord)
    assert rows[0].dataset_id == "ds006072"
    assert rows[0].version == "1.1.0"
    assert rows[0].doi == "10.18112/openneuro.ds006072.v1.1.0"
    assert rows[0].modalities == ["MRI"]


def test_neuromaps_catalog_joins_registry_metadata_and_license(tmp_path, monkeypatch):
    broker = ScientificDataBroker(tmp_path)
    files = [{
        "source": "beliveau2017", "desc": "cimbi36", "space": "MNI152",
        "res": "1mm", "format": "volume", "fname": "map.nii.gz",
        "rel_path": "beliveau2017/cimbi36/MNI152", "checksum": "abc",
        "tags": ["receptors", "PET"], "url": ["4mw3a", "file-id"],
    }]
    metadata = [{
        "annot": {
            "source": "beliveau2017", "desc": "cimbi36", "space": "MNI152",
            "res": "1mm",
        },
        "full_desc": "PET atlas of the serotonin 5-HT2A receptor",
        "refs": {
            "primary": [{"citation": "Beliveau et al. 2017", "bibkey": "b"}],
            "secondary": [],
        },
        "demographics": {"N": 29},
        "license": {"type": "CC BY-NC-SA 4.0"},
    }]
    monkeypatch.setattr(
        broker, "_neuromaps_registry", lambda: ("0.0.7", files, metadata))
    rows = broker._discover_neuromaps("psychedelic serotonin receptor atlas", 5)
    assert rows[0].dataset_id == "beliveau2017-cimbi36-MNI152-1mm"
    assert rows[0].license == "CC BY-NC-SA 4.0"
    assert rows[0].metadata["demographics"]["N"] == 29
    assert rows[0].modalities == ["PET"]


def test_acquisition_locks_plan_hashes_files_and_materializes(tmp_path, monkeypatch):
    cfg = SimpleNamespace(
        research_data_auto=True,
        research_data_max_gb=1.0,
        research_data_reserve_gb=0.0,
        research_data_file_limit=20,
        research_data_timeout=30,
    )
    broker = ScientificDataBroker(tmp_path / "broker", cfg=cfg)
    content = b"region,value\nA,1\nB,2\n"

    def resolve(request):
        return ({
            "source": "openneuro",
            "dataset_id": "ds000001",
            "title": "Tiny fixture",
            "version": "1.0.0",
            "doi": "10.example/data",
            "license": "CC0",
            "citation": "10.example/data",
            "species": "human",
            "modalities": ["MRI"],
            "include": ["values.csv"],
            "file_count": 1,
            "bytes": len(content),
        }, [
            DataFile(
                path="values.csv", url="https://s3.amazonaws.com/openneuro.org/x",
                bytes=len(content)),
        ])

    def download(source, item, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        return {
            "path": str(destination), "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "etag": "", "url": item.url, "cache_hit": False,
        }

    monkeypatch.setattr(broker, "_resolve", resolve)
    monkeypatch.setattr(broker, "_download", download)
    target = tmp_path / "certificate" / "_data"
    report = broker.acquire_many(
        [{
            "source": "openneuro", "id": "ds000001", "version": "1.0.0",
            "alias": "primary", "include": ["values.csv"], "species": "human",
        }],
        plan=_plan(),
        alignment=_alignment(),
        materialize_to=target,
    )
    assert report["confirmatory_ready"]
    assert report["provenance_complete"]
    assert len(report["plan_hash"]) == 64
    assert (target / "primary" / "values.csv").read_bytes() == content
    assert Path(report["manifest"]).is_file()


def test_data_workbench_strength_needs_confirmatory_manifest(tmp_path):
    manifest = tmp_path / "manifest.json"
    base = {
        "validation_evidence": {"computationally_reproduced": True},
        "data_evidence": {
            "not_applicable": False, "confirmatory_ready": False,
            "result_summary_ready": True,
        },
    }
    manifest.write_text(json.dumps(base), encoding="utf-8")
    claim = {"manifest": str(manifest)}
    assert ResearchLoop._workbench_strength(claim, True) == "executable"
    base["data_evidence"]["confirmatory_ready"] = True
    manifest.write_text(json.dumps(base), encoding="utf-8")
    assert ResearchLoop._workbench_strength(claim, True) == "computational"


def test_research_graph_includes_typed_dataset_frontier():
    from spiral.research_graph import build_graph_data

    graph = build_graph_data({
        "topic": "multimodal imaging",
        "data_catalog": {
            "records": [{
                "source": "openneuro", "dataset_id": "ds006072",
                "title": "Psilocybin mapping", "version": "1.1.0",
                "license": "CC0", "modalities": ["MRI"],
                "url": "https://openneuro.org/datasets/ds006072",
            }],
        },
    })
    assert any(node["type"] == "dataset" for node in graph["nodes"])
    assert any(edge["type"] == "data" for edge in graph["edges"])
