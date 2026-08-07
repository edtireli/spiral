"""Source adapters — all offline; every provider is faked, no network ever."""
import json

from spiral.sources import (
    Record, biorxiv, crossref, europepmc, jats_to_text, medrxiv, pubmed, unpaywall,
)


def _epmc_payload():
    return {"resultList": {"result": [
        {"id": "PPR123", "source": "PPR", "doi": "10.1101/2023.01.01.522334",
         "title": "TDP-43 aggregation in cortical neurons",
         "publisher": "Cold Spring Harbor Laboratory",
         "abstractText": "We show that TDP-43 aggregation increases with age.",
         "firstPublicationDate": "2023-01-02", "isOpenAccess": "N",
         "authorList": {"author": [{"fullName": "A Bianchi"}, {"fullName": "B Kol"}]}},
        {"id": "PPR456", "source": "PPR", "doi": "10.1101/2023.02.02.400000",
         "title": "A clinical trial protocol for stroke",
         "publisher": "Cold Spring Harbor Laboratory Press - medRxiv",
         "abstractText": "Protocol.", "firstPublicationDate": "2023-02-02"},
        {"id": "PMC9", "source": "PMC", "pmcid": "PMC9999999", "inEPMC": "Y",
         "isOpenAccess": "Y", "doi": "10.1000/pmcopen",
         "title": "Open access neuroscience review",
         "abstractText": "A review.", "pmid": "34000000",
         "authorList": {"author": [{"firstName": "C", "lastName": "Vega"}]}},
    ]}}


def test_europepmc_normalises_sources_and_uids():
    seen = {}
    def fake_json(url, timeout):
        seen["url"] = url
        return _epmc_payload()
    rep = {}
    recs = europepmc("tdp-43 aggregation", k=6, report=rep, fetch_json=fake_json)
    assert rep["source_ok"] and rep["result_count"] == 3
    byuid = {r.uid: r for r in recs}
    # DOI is the join key when present
    assert "doi:10.1101/2023.01.01.522334" in byuid
    bio = byuid["doi:10.1101/2023.01.01.522334"]
    assert bio.source == "biorxiv" and bio.authors == ["A Bianchi", "B Kol"]
    med = byuid["doi:10.1101/2023.02.02.400000"]
    assert med.source == "medrxiv"
    pmc = byuid["doi:10.1000/pmcopen"]
    assert pmc.source == "pmc" and pmc.full_text_kind == "jats"
    # JATS is served by NCBI efetch (db=pmc, numeric id) — the reliable full-text route
    assert "efetch.fcgi?db=pmc&id=9999999" in pmc.full_text_url


def test_biorxiv_and_medrxiv_filter():
    def fake_json(url, timeout):
        return _epmc_payload()
    assert [r.source for r in biorxiv("x", k=3, fetch_json=fake_json)] == ["biorxiv"]
    assert [r.source for r in medrxiv("x", k=3, fetch_json=fake_json)] == ["medrxiv"]


def test_europepmc_down_degrades_not_raises():
    def boom(url, timeout):
        raise RuntimeError("EBI 503")
    rep = {}
    assert europepmc("x", report=rep, fetch_json=boom) == []
    assert rep["source_ok"] is False and "503" in rep["error"]


def test_pubmed_three_call_flow_and_doi_join():
    calls = []
    def fake_json(url, timeout):
        calls.append(url)
        if "esearch" in url:
            return {"esearchresult": {"idlist": ["34567890", "34567891"]}}
        return {"result": {
            "34567890": {"title": "Neural correlates of memory",
                         "articleids": [{"idtype": "doi", "value": "10.1016/j.neuron.2021.01.001"}],
                         "authors": [{"name": "D Ross"}], "pubdate": "2021",
                         "fulljournalname": "Neuron"},
            "34567891": {"title": "No DOI here", "articleids": [],
                         "authors": [], "pubdate": "2020"}}}
    def fake_text(url, timeout):
        return "Abstract one about memory.\n\n\nAbstract two."
    rep = {}
    recs = pubmed("memory", k=2, report=rep, fetch_json=fake_json, fetch_text=fake_text)
    assert rep["source_ok"] and len(recs) == 2
    assert recs[0].uid == "doi:10.1016/j.neuron.2021.01.001"   # DOI preferred
    assert recs[1].uid == "pmid:34567891"                       # falls back to PMID
    assert recs[0].venue == "Neuron"
    assert any("esummary" in c for c in calls)


def test_pubmed_empty_is_clean():
    def fake_json(url, timeout):
        return {"esearchresult": {"idlist": []}}
    rep = {}
    assert pubmed("nothing", report=rep, fetch_json=fake_json,
                  fetch_text=lambda u, t: "") == []
    assert rep["source_ok"] and rep["result_count"] == 0


def test_crossref_requires_doi_and_extracts_pdf_link():
    def fake_json(url, timeout):
        return {"message": {"items": [
            {"DOI": "10.1021/jacs.1c00001", "title": ["A chemistry paper"],
             "author": [{"given": "E", "family": "Frank"}],
             "published": {"date-parts": [[2022, 5]]},
             "abstract": "<jats:p>We synthesize.</jats:p>",
             "container-title": ["JACS"],
             "link": [{"content-type": "application/pdf", "URL": "https://x/y.pdf"}]},
            {"title": ["No DOI, dropped"]}]}}
    rep = {}
    recs = crossref("catalysis", report=rep, fetch_json=fake_json)
    assert len(recs) == 1 and recs[0].uid == "doi:10.1021/jacs.1c00001"
    assert recs[0].full_text_url == "https://x/y.pdf" and recs[0].full_text_kind == "pdf"
    assert recs[0].abstract == "We synthesize."      # jats tags stripped


def test_unpaywall_resolves_best_oa_pdf():
    def fake_json(url, timeout):
        assert "10.1101" in url
        return {"best_oa_location": {"url_for_pdf": "https://oa/paper.pdf"}}
    assert unpaywall("https://doi.org/10.1101/abc", fetch_json=fake_json) == (
        "https://oa/paper.pdf", "pdf")
    assert unpaywall("", fetch_json=fake_json) == ("", "")


def test_jats_extracts_body_drops_refs_keeps_math():
    xml = """<article xmlns:mml="http://www.w3.org/1998/Math/MathML">
      <front><article-meta><abstract><p>We model kinetics.</p></abstract></article-meta></front>
      <body>
        <sec><title>Results</title>
          <p>The rate constant is <tex-math>k = 3.1</tex-math> per second.</p></sec>
      </body>
      <back><ref-list><ref><mixed-citation>Smith 2020, do not include</mixed-citation></ref></ref-list></back>
    </article>"""
    text = jats_to_text(xml)
    assert "We model kinetics" in text and "Results" in text
    assert "$k = 3.1$" in text
    assert "do not include" not in text          # references dropped


def test_jats_malformed_falls_back_to_tag_strip():
    assert "hello world" in jats_to_text("<p>hello <b>world</b> <ref>drop")


# ---------------------------------------------------------------- cite graph (bio-safe)
def test_parse_edges_survives_null_data():
    """S2 returns {"data": null} for some papers — this crashed a live bio run."""
    from spiral.cite_graph import parse_edges
    assert parse_edges({"data": None}, "references") == []
    assert parse_edges(None, "citations") == []
    assert parse_edges({"data": [None, "junk", {}]}, "references") == []


def test_parse_edges_keeps_doi_pmid_pmc_edges():
    """Bio/med neighbours carry DOI/PMID/PMCID, not ArXiv — they must snowball too."""
    from spiral.cite_graph import parse_edges
    payload = {"data": [
        {"citedPaper": {"externalIds": {"DOI": "10.1101/2023.01.01.522"}, "title": "a", "citationCount": 5}},
        {"citedPaper": {"externalIds": {"PubMed": "34567890"}, "title": "b"}},
        {"citedPaper": {"externalIds": {"PubMedCentral": "9999999"}, "title": "c"}},
        {"citedPaper": {"externalIds": {"ArXiv": "2401.01234"}, "title": "d"}},
        {"citedPaper": {"externalIds": {}, "title": "no id"}},
    ]}
    edges = {e.arxiv_id: (e.source, e.doi) for e in parse_edges(payload, "references")}
    assert edges["doi:10.1101/2023.01.01.522"] == ("biorxiv", "10.1101/2023.01.01.522")
    assert edges["pmid:34567890"] == ("pubmed", "")
    assert edges["pmc:PMC9999999"] == ("pmc", "")
    assert edges["2401.01234"] == ("arxiv", "")
    assert len(edges) == 4      # id-less dropped


def test_s2_id_scheme_mapping():
    from spiral.cite_graph import _s2_id
    assert _s2_id("doi:10.1101/x") == "DOI:10.1101/x"
    assert _s2_id("pmid:1") == "PMID:1"
    assert _s2_id("pmc:PMC9") == "PMCID:PMC9"
    assert _s2_id("2401.01234v2") == "arXiv:2401.01234"
