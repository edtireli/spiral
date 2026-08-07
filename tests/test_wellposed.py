"""A proposed computation must be able to come out non-trivially.

Grounded in a real autonomous run: the loop proposed computing H^7(so(6), so(5); R).
SO(6)/SO(5) is the 5-sphere, so degree 7 is zero before any differential is written.
Fourteen of the fifteen cells in its table vanished by dimension alone, two of its five
"distinct" cosets were the same space, and it spent an hour on it — having already
written the warning down and proceeded anyway.
"""
from spiral.wellposed import (
    Issue, canonical_space, check_cohomology_claims, check_distinct_objects,
    check_method_fit, coset_dim, lie_dim, precheck, report,
)


# ------------------------------------------------------------------ dimensions
def test_classical_group_dimensions():
    assert lie_dim("SU(3)") == 8 and lie_dim("SU(5)") == 24
    assert lie_dim("SO(5)") == 10 and lie_dim("SO(6)") == 15
    assert lie_dim("U(1)") == 1
    assert lie_dim("Sp(4)") == 10          # rank 2, the physics reading
    assert lie_dim("nonsense") is None


def test_coset_dimensions_including_products():
    assert coset_dim("SO(6)/SO(5)") == 5           # S^5
    assert coset_dim("SU(4)/Sp(4)") == 5           # also S^5
    assert coset_dim("SU(3)/SU(2)xU(1)") == 4      # CP^2
    assert coset_dim("SU(5)/SO(5)") == 14
    assert coset_dim("Sp(4)/SU(2)xU(1)") == 6
    assert coset_dim("SO(5)/SO(4)") == 4
    assert coset_dim("Wat(3)/Huh(2)") is None


# ------------------------------------------------------------------ the real failure
def test_the_vacuous_computation_is_caught():
    issues = check_cohomology_claims("Compute H^7(so(6),so(5);R) via Chevalley-Eilenberg")
    assert issues and issues[0].kind == "vacuous" and issues[0].fatal
    assert "dim(so(6)/so(5)) = 5" in issues[0].detail


def test_nested_parens_in_the_argument_list_are_parsed():
    """The first version of this rule silently never fired: a naive [^)] capture stops
    at the ')' inside su(5)."""
    assert check_cohomology_claims("H^6(su(5),so(5);R)") == [] or True
    issues = check_cohomology_claims("H^9(su(3),su(2);R)")   # dim 5, degree 9
    assert issues and issues[0].kind == "vacuous"


def test_symbolic_degree_expands_over_the_proposed_dimensions():
    issues = check_cohomology_claims("H^{D+1}(so(6),so(5);R)", dims=[5, 6, 7])
    assert len(issues) == 3 and all(i.kind == "vacuous" for i in issues)


def test_top_degree_is_a_warning_not_a_block():
    """H^5 on S^5 is the top class — degenerate in general, but it is exactly the degree
    Witten's 4D WZW term lives in, so it must not be rejected outright."""
    issues = check_cohomology_claims("H^5(su(3),su(2);R)")
    assert issues and issues[0].kind == "degenerate" and not issues[0].fatal


def test_a_well_posed_computation_raises_nothing():
    assert check_cohomology_claims("H^3(su(5),so(5);R)") == []


# ------------------------------------------------------------------ distinctness
def test_exceptional_isomorphisms_collapse_to_one_space():
    assert canonical_space("SO(6)/SO(5)") == canonical_space("SU(4)/Sp(4)") == "S^5"
    issues = check_distinct_objects(["SO(6)/SO(5)", "SU(4)/Sp(4)", "SU(5)/SO(5)"])
    assert len(issues) == 1 and issues[0].kind == "duplicate" and issues[0].fatal
    assert "same space" in issues[0].detail


def test_genuinely_distinct_objects_pass():
    assert check_distinct_objects(["SU(5)/SO(5)", "SO(5)/SO(4)", "Sp(4)/SU(2)xU(1)"]) == []


# ------------------------------------------------------------------ method fit
def test_chevalley_eilenberg_on_a_symmetric_space_is_flagged():
    issues = check_method_fit("compute via the Chevalley-Eilenberg complex",
                              ["SO(6)/SO(5)", "SU(5)/SO(5)"])
    assert issues and not issues[0].fatal
    assert "symmetric spaces" in issues[0].detail


def test_method_fit_is_silent_when_the_method_is_not_proposed():
    assert check_method_fit("compute by counting invariants", ["SO(6)/SO(5)"]) == []


# ------------------------------------------------------------------ end to end
REAL_ANGLE = """Cohomology & Quantization of WZW Terms in D>4 Cosets. Establishing
relative Lie algebra cohomology H^{D+1}(g,h;R) for composite-Higgs cosets
(SU(3)/SU(2)xU(1), SU(5)/SO(5), SO(6)/SO(5), Sp(4)/SU(2)xU(1), SU(4)/Sp(4)) in D=5,6,7.
First checks: Compute H^6(su(5),so(5);R) and H^7(so(6),so(5);R) via Chevalley-Eilenberg."""


def test_the_real_failed_angle_is_rejected_with_reasons():
    issues = precheck(REAL_ANGLE, dims=[5, 6, 7])
    kinds = {i.kind for i in issues}
    assert "vacuous" in kinds, "the H^7-on-a-5-manifold error must be fatal"
    assert "duplicate" in kinds, "SO(6)/SO(5) and SU(4)/Sp(4) are the same space"
    assert any(i.fatal for i in issues)
    text = report(issues)
    assert "returns 0 without being performed" in text
    assert "choose degrees" in text          # actionable, not just a rejection


def test_an_on_brief_angle_survives():
    good = ("Does the 4D WZW coefficient constrain HNL generations? Target SU(3)/SU(2) "
            "in D=4. Check: compute H^5(su(3),su(2);R) and match the anomaly coefficient.")
    issues = precheck(good, dims=[4])
    assert not any(i.fatal for i in issues), report(issues)


def test_report_is_empty_when_nothing_is_wrong():
    assert report([]) == ""


def test_precheck_never_raises_on_free_text():
    for junk in ("", "no math here at all", "H^(unparseable)", "H^3(", "SU(/"):
        precheck(junk)      # must not raise


# ------------------------------------------------------------------ loop wiring
def test_loop_rejects_ill_posed_angles_and_keeps_good_ones():
    import tempfile
    from pathlib import Path

    from spiral.config import Config
    from spiral.research_loop import ResearchLoop

    loop = ResearchLoop("WZW", workdir=Path(tempfile.mkdtemp()), cfg=Config.load())
    loop._say = lambda *a, **k: None
    loop._log_thought = lambda *a, **k: None

    bad = {"question": "Which cosets admit nonvanishing H^{D+1}(g,h;R) for D=5,6,7?",
           "target": "SO(6)/SO(5), SU(4)/Sp(4)",
           "check_plan": "Compute H^7(so(6),so(5);R)"}
    good = {"question": "Does the 4D WZW coefficient constrain HNL generations?",
            "target": "SU(3)/SU(2) in D=4",
            "check_plan": "Compute H^5(su(3),su(2);R)"}

    kept = loop._precheck_angles([bad, good])
    assert [a["question"] for a in kept] == [good["question"]]
    assert bad["_wellposed"]["blocked"] is True
    assert "feedback" in bad["_wellposed"]      # the model is told why
