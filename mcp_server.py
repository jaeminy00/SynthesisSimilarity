"""Literature-based precursor recommender MCP server.

Architecture: PrecursorSelector (He et al. Sci. Adv. 2023).
Weights: v0.5 — retrained on the Lee text-mined dataset (66,154 solid-state
reactions, 46,816 papers); benchmark parity with the published model.
Knowledge base: the full converted Lee dataset (28,844 unique targets).
Unavailable-precursor default: compounds-only list (elemental precursors allowed).
Conditions + impurity phases: rsc_impurity/literature.db (build_literature_db.py,
see LITERATURE_DB.md), one row per knowledge-base reaction, counted by paper.

Run:      .venv/bin/python mcp_server.py
Selftest: .venv/bin/python mcp_server.py --selftest
uvx:      SYNSIM_DATA_DIR=/path/to/rsc/and/rsc_impurity uvx --python 3.11 --from
          git+https://github.com/jaeminy00/SynthesisSimilarity@feature/impurity
          synthesis-similarity-mcp
"""
import collections
import contextlib
import functools
import json
import math
import os
import sqlite3
import statistics
import sys

import numpy as np
from mcp.server.fastmcp import FastMCP
from pymatgen.core import Composition

from SynthesisSimilarity import PrecursorsRecommendation
from SynthesisSimilarity.core.utils import formula_to_array, array_to_formula
from SynthesisSimilarity.core.mat_featurization import featurize_list_of_composition

mcp = FastMCP("synthesis-similarity")

REPO = os.path.dirname(os.path.abspath(__file__))
# rsc/ and rsc_impurity/ are data, not code: an installed server (uvx) needs their path
DATA = os.environ.get("SYNSIM_DATA_DIR", os.path.join(REPO, "SynthesisSimilarity"))

print("loading v0.5 model + 66k-reaction knowledge base (~25s)...", file=sys.stderr)
# The library and Keras print to stdout while loading, but stdout is the JSONRPC
# wire — stray lines there break the client's handshake. Park stdout on stderr
# until the model is up; once serving, the SDK diverts fd 1 to stderr itself.
with contextlib.redirect_stdout(sys.stderr):
    REC = PrecursorsRecommendation(
        model_dir=os.path.join(REPO, "SynthesisSimilarity/models/v05"),
        freq_path=os.path.join(DATA, "rsc/pre_count_normalized_by_rxn_ss.json"),
        data_path=os.path.join(DATA, "rsc_impurity/ss_rxns_kb.npz"),
        path_pres_ref=os.path.join(DATA, "rsc/pres_name_ref.json"),
        path_pres_unavail=os.path.join(
            REPO, "SynthesisSimilarity/rsc/pres_unavail_compounds.json"
        ),
        all_to_knowledge_base=True,
    )
# raw_index -> reaction metadata (doi/year/operations), for provenance lookups
RAW = {}
for _r in REC.train_reactions:
    RAW[_r["raw_index"]] = _r


def _embed(formulas):
    comps = [formula_to_array(f, REC.all_elements) for f in formulas]
    feats = featurize_list_of_composition(
        comps=comps, ele_order=REC.all_elements, featurizer_type=REC.featurizer_type
    )
    vecs = REC.framework_model.get_mat_vector(np.array(feats)).numpy()
    return vecs / np.linalg.norm(vecs, axis=-1, keepdims=True)


def _check_formula(formula):
    """Return error string, or None if the formula is usable."""
    try:
        comp = formula_to_array(formula, REC.all_elements)
    except Exception as e:
        return f"cannot parse formula '{formula}': {e}"
    if not np.any(comp):
        return (
            f"'{formula}' contains no element in the model's "
            f"{len(REC.all_elements)}-element vocabulary"
        )
    return None


def _normalize(formula):
    """Input formula -> the knowledge base's canonical formula string."""
    return array_to_formula(formula_to_array(formula, REC.all_elements), REC.all_elements)


LIT_DB = os.path.join(DATA, "rsc_impurity/literature.db")
VOLATILE = {"C", "H", "N", "O"}  # leave as CO2/H2O/NOx/O2; ignored when classifying phases
# ponytail: fixed evidence thresholds; tune once labels are checked against real verdicts
RARE_UPPER = 0.15     # 0 papers report a phase and the 95% upper bound is below this -> "rarely reported"
MISSED_LOWER = 0.10   # literature phase whose 95% lower bound reaches this -> "missed by simulation"
STOICH_TOL = 0.02     # cation-ratio deviation tolerated as "balanced"


def _lit():
    return contextlib.closing(sqlite3.connect(f"file:{LIT_DB}?mode=ro", uri=True))


# raw_index -> max heating temperature (deg C; K converted, implausible values dropped)
LIT_T = {}
if os.path.exists(LIT_DB):
    with _lit() as _con:
        LIT_T = dict(_con.execute("SELECT raw_index, max_temp_C FROM reactions"))
else:
    print(f"WARNING: {LIT_DB} missing; run build_literature_db.py. "
          "Impurity/condition fields are unavailable.", file=sys.stderr)


def _max_heating_T(raw_index):
    return LIT_T.get(raw_index)


def _norm_or_raw(formula):
    try:
        return _normalize(formula)
    except Exception:
        return formula


def _route_rows(target_key, precursor_key=None):
    """literature.db reactions for a target (optionally one precursor set), with their named phases."""
    where, params = "r.target = ?", [target_key]
    if precursor_key is not None:
        where, params = where + " AND r.precursor_key = ?", params + [precursor_key]
    with _lit() as con:
        con.row_factory = sqlite3.Row
        rows = {r["raw_index"]: {**dict(r), "phases": []} for r in con.execute(
            f"SELECT r.* FROM reactions r WHERE {where}", params)}
        for i, f, raw in con.execute(
            f"SELECT i.raw_index, i.formula, i.formula_raw FROM impurities i "
            f"JOIN reactions r USING (raw_index) WHERE {where}", params):
            rows[i]["phases"].append(f or raw)
    return list(rows.values())


def _by_paper(rows):
    """Group reactions into papers (distinct DOI; a reaction without a DOI counts as its own paper).
    A paper reports a phase if any of its reactions does, so a 5-temperature study counts once."""
    papers = {}
    for r in rows:
        p = papers.setdefault(r["doi"] or f"raw_index:{r['raw_index']}",
                              {"phases": set(), "unidentified": False, "temps": []})
        p["phases"].update(r["phases"])
        p["unidentified"] |= r["impurity_state"] == "unidentified"
        if r["max_temp_C"] is not None:
            p["temps"].append(r["max_temp_C"])
    return papers


def _paper_states(papers):
    named = sum(1 for p in papers.values() if p["phases"])
    unid = sum(1 for p in papers.values() if not p["phases"] and p["unidentified"])
    return {"papers_none_reported": len(papers) - named - unid,
            "papers_unidentified_only": unid, "papers_with_named_phases": named}


def _stats(values):
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    return {"n": len(v), "min": round(v[0], 1), "median": round(statistics.median(v), 1),
            "max": round(v[-1], 1)}


def _cations(formula):
    try:
        return {e.symbol for e in Composition(formula).elements} - VOLATILE
    except Exception:
        return None


def _classify(phase, precursors):
    """leftover precursor: every non-C/H/N/O element comes from one precursor (unreacted or
    decomposed, e.g. CaO from CaCO3); contamination: an element no precursor supplies;
    otherwise a secondary phase (elements from 2+ precursors, e.g. Li2SiO3)."""
    els = _cations(phase)
    if els is None:
        return "unparseable"
    pres = [s for s in map(_cations, precursors) if s is not None]
    if not els <= set().union(*pres):
        return "contamination"
    if any(els <= s for s in pres):
        return "leftover precursor"
    return "secondary phase"


def _wilson(k, n, z=1.96):
    """95% Wilson score interval for k of n papers."""
    if n == 0:
        return None
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def _balance(target, precursors):
    """Balanced reaction (library tuple: target, {'left','right'}, subs, equation) or None."""
    try:
        rxn = REC.get_reaction(
            target_formula=target, precursors_formulas=precursors, ref_materials_comp={}
        )
    except Exception:
        rxn = None
    if rxn is None:
        from SynthesisSimilarity.scripts_utils.reaction_utils import balance_w_rxn_network

        try:
            rxn = balance_w_rxn_network(
                target_formula=target, precursors_formulas=precursors, ref_materials_comp={}
            )
        except Exception:
            rxn = None
    return rxn


def _balanced_equation(target, precursors, rows):
    """Dataset's balanced target equation for this route if any paper has one, else synthesis-similarity's."""
    eqs = collections.Counter((r["equation"], r["coefficients"]) for r in rows if r["equation"])
    if eqs:
        (eq, coeffs), _ = eqs.most_common(1)[0]
        return {"equation": eq, "source": "dataset", "left": json.loads(coeffs)["left"] if coeffs else None}
    rxn = _balance(target, precursors)
    if rxn is None:
        return {"equation": None, "source": None, "left": None}
    try:
        left = {f: float(c) for f, c in (rxn[1].get("left") or {}).items()}
    except Exception:
        left = None
    return {"equation": rxn[3], "source": "synthesis-similarity", "left": left}


def _stoichiometry(target, precursors, reactant_amounts, balanced_left):
    """Compare a simulated mix's non-C/H/N/O element ratios with the target's.
    Off-stoichiometry runs are data points: they are flagged, not discarded."""
    try:
        mix = sum((Composition(f) * float(a) for f, a in reactant_amounts.items()), Composition())
        tgt = Composition(target)
    except Exception as e:
        return {"status": "error", "reason": f"cannot parse reactant_amounts: {e}"}
    t = {e.symbol: a for e, a in tgt.items() if e.symbol not in VOLATILE}
    m = {e.symbol: a for e, a in mix.items() if e.symbol not in VOLATILE}
    out = {"extra_elements": sorted(set(m) - set(t))}  # flux / dopant / additive
    missing = sorted(set(t) - set(m))
    if missing:
        out.update(status="off_stoichiometry", missing_elements=missing)
    else:
        ratio = {e: m[e] / t[e] for e in t}
        lim = min(ratio, key=ratio.get)
        excess = {e: round((r / ratio[lim] - 1) * 100, 1) for e, r in ratio.items()
                  if r / ratio[lim] - 1 > STOICH_TOL}
        out.update(status="off_stoichiometry" if excess else "balanced",
                   limiting_element=lim if excess else None, excess_pct_vs_limiting=excess)
    if out["status"] == "off_stoichiometry":
        names = {_norm_or_raw(p): p for p in precursors}  # report in the caller's spelling
        out["balanced_reactant_amounts"] = (
            {names[_norm_or_raw(f)]: c for f, c in balanced_left.items() if _norm_or_raw(f) in names}
            if balanced_left else None
        )
        out["action"] = ("Report this run as an off-stoichiometry data point, and also simulate "
                         "the balanced recipe.")
    return out


def _recipe_sources(recipe, pres_set=None):
    """DOIs/years/temps behind a knowledge-base recipe (optionally one precursor set)."""
    if pres_set is not None:
        indices = recipe["pres_raw_index"].get(pres_set, [])
    else:
        indices = list(recipe["raw_index"])
    out = []
    for i in indices[:10]:
        r = RAW.get(i)
        if r:
            out.append(
                {"doi": r.get("doi"), "year": r.get("year"), "max_heating_C": _max_heating_T(i)}
            )
    return out


def _format_provenance(prov):
    """Format the exact provenance recorded by recommend_precursors_by_similarity."""
    if prov is None:
        return {
            "reference_material": None,
            "note": "assembled from the most common precursor per element; "
            "no single literature precedent",
        }
    recipe = REC.train_targets_recipes[prov["ref_index"]]
    ref_set = tuple(prov["ref_set"])
    return {
        "reference_material": REC.train_targets_formulas[prov["ref_index"]],
        "similarity": round(prov["similarity"], 3),
        "reported_set": prov["ref_set"],
        "n_papers": recipe["pres"].get(ref_set, 0),
        "added_beyond_reference": prov["added"],
        "sources": _recipe_sources(recipe, ref_set)[:3],
    }



def _quiet(fn):
    """The library prints to stdout inside tool calls (e.g. precursors_recommendation_utils
    prints len(test_targets_formulas) on every recommend_precursors). stdout is the JSONRPC
    wire, so a stray line corrupts the stream and the client errors on teardown. Park stdout
    on stderr for the duration of every tool call."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with contextlib.redirect_stdout(sys.stderr):
            return fn(*args, **kwargs)
    return wrapper


@mcp.tool()
@_quiet
def recommend_precursors(
    target: str, top_n: int = 5, avoid: list[str] | None = None, validate: bool = True
) -> dict:
    """Recommend precursor sets for a target material using literature-learned
    synthesis similarity (PrecursorSelector architecture, retrained on the Lee
    dataset). Each set includes a balanced reaction (when validate=True) and the
    literature precedent it derives from. `avoid`: precursor formulas to exclude
    (merged with a built-in compound blacklist; elemental precursors are allowed)."""
    if err := _check_formula(target):
        return {"error": err}
    not_avail = REC.pre_set_unavail_default | set(avoid or [])
    preds = REC.recommend_precursors(
        target_formula=[target],
        top_n=top_n,
        validate_reaction=validate,
        precursors_not_available=not_avail,
    )[0]
    results = []
    if validate:
        for p in preds:
            results.append(
                {
                    "precursors": list(p["precursors"]),
                    "reaction": p["reaction_string"],
                    "precedent": _format_provenance(p["provenance"]),
                }
            )
    else:
        for pres, prov in zip(preds["precursors_predicts"], preds["provenance"]):
            results.append(
                {
                    "precursors": list(pres),
                    "reaction": None,
                    "precedent": _format_provenance(prov),
                }
            )
    return {
        "target": target,
        "recommendations": results,
        "summary": f"{len(results)} precursor set(s) recommended for {target}",
    }


@mcp.tool()
@_quiet
def find_similar_materials(formula: str, k: int = 10) -> dict:
    """Find the k most synthesis-similar known materials in the text-mined knowledge
    base (28.8k unique targets from 66,154 solid-state reactions across ~47k papers).
    Similarity is the cosine of learned composition embeddings: materials made via
    similar precursors score high. Returns each neighbor's most common precursor
    set and paper count."""
    if err := _check_formula(formula):
        return {"error": err}
    sims = (_embed([formula]) @ REC.train_targets_vecs.T)[0]
    neighbors = []
    for i in np.argsort(sims)[::-1][:k]:
        recipe = REC.train_targets_recipes[i]
        best_set, best_count = recipe["pres"].most_common(1)[0]
        neighbors.append(
            {
                "formula": REC.train_targets_formulas[i],
                "similarity": round(float(sims[i]), 3),
                "n_papers": sum(recipe["pres"].values()),
                "synthesis_types": dict(recipe["syn_type"]),
                "most_common_precursors": list(best_set),
                "precursor_set_count": best_count,
            }
        )
    return {"query": formula, "neighbors": neighbors}


@mcp.tool()
@_quiet
def get_literature_recipes(target: str) -> dict:
    """Look up every precursor set reported in the literature for a known target
    material (exact composition match). Per precursor set: n_reactions and n_papers
    (distinct DOIs), max heating temperature (deg C) and longest heating step (h)
    statistics over ALL its reactions, an impurity summary counted by paper (papers
    with none reported / unidentified phase only / named phases, and the most-reported
    phases classed as leftover precursor, secondary phase or contamination), the
    dataset's balanced equation, and up to 10 sources (DOI, year, max heating C).
    Use find_similar_materials first if the target might not be in the knowledge
    base; use route_impurities for one route in detail."""
    if err := _check_formula(target):
        return {"error": err}
    key = _normalize(target)
    recipe = REC.train_targets.get(key)
    if recipe is None:
        return {
            "error": f"'{target}' (normalized: {key}) not in knowledge base; "
            "use find_similar_materials for nearby known targets"
        }
    rows = _route_rows(key) if LIT_T else []
    by_set = collections.defaultdict(list)
    for r in rows:
        by_set[r["precursor_key"]].append(r)
    recipes = []
    for pres_set, count in recipe["pres"].most_common():
        entry = {"precursors": list(pres_set), "n_reactions": count}
        rs = by_set.get("+".join(pres_set), [])
        if rs:
            papers = _by_paper(rs)
            phases = collections.Counter(f for p in papers.values() for f in p["phases"])
            eqs = collections.Counter(r["equation"] for r in rs if r["equation"])
            entry.update(
                n_papers=len(papers),
                max_heating_C=_stats(r["max_temp_C"] for r in rs),
                longest_heating_h=_stats(r["max_time_h"] for r in rs),
                impurities={
                    **_paper_states(papers),
                    "most_reported": [
                        {"formula": f, "class": _classify(f, pres_set), "papers": k}
                        for f, k in phases.most_common(5)
                    ],
                },
                balanced_equation=eqs.most_common(1)[0][0] if eqs else None,
            )
        entry["sources"] = _recipe_sources(recipe, pres_set)
        recipes.append(entry)
    return {
        "target": key,
        "n_reactions": len(recipe["raw_index"]),
        "n_papers": len(_by_paper(rows)) if rows else None,
        "synthesis_types": dict(recipe["syn_type"]),
        "recipes": recipes,
    }


@mcp.tool()
@_quiet
def route_impurities(
    target: str,
    precursors: list[str],
    simulated_phases: list[str] | None = None,
    reactant_amounts: dict[str, float] | None = None,
) -> dict:
    """Literature impurity evidence for ONE route (target + precursor set), counted by
    paper (distinct DOI): a paper reports a phase if any of its reactions does.
    Phases are split into leftover precursors (unreacted or decomposed precursor, e.g.
    CaO from CaCO3: incomplete conversion), secondary phases (elements from 2+ precursors,
    e.g. Li2SiO3: a competing phase) and contamination (an element no precursor supplies,
    e.g. Al from a crucible). Each phase: papers reporting it / papers on the route, 95%
    Wilson interval, and median max heating temperature in papers with vs without it.
    "None reported" means the text-mined paper lists no impurity, not proof of purity.

    simulated_phases: non-target phases a simulation (e.g. ReactCA) predicts. Each
    secondary phase is labeled confirmed (>=1 paper reports it), rarely reported (0 papers
    and the interval's upper bound < 15%), inconclusive (0 papers, too few to say), or no
    literature; common literature secondary phases the simulation lacks are listed as missed.
    reactant_amounts: molar amounts the simulation used, checked against the balanced
    equation (dataset, else synthesis-similarity). Off-stoichiometry runs are still
    compared and reported, flagged, with the balanced amounts to simulate as well."""
    if not LIT_T:
        return {"error": "literature.db missing; run build_literature_db.py"}
    for f in [target, *precursors]:
        if err := _check_formula(f):
            return {"error": err}
    tkey = _normalize(target)
    pres = sorted({_normalize(p) for p in precursors})
    rows = _route_rows(tkey, "+".join(pres))
    papers = _by_paper(rows)
    n = len(papers)
    counts = collections.Counter(f for p in papers.values() for f in p["phases"])

    def phase_stats(f, k):
        temps = lambda has: [statistics.median(p["temps"]) for p in papers.values()
                             if (f in p["phases"]) == has and p["temps"]]
        with_, without = temps(True), temps(False)
        return {"formula": f, "papers": k, "of": n, "ci95": _wilson(k, n),
                "median_max_heating_C": {
                    "with": round(statistics.median(with_)) if with_ else None,
                    "without": round(statistics.median(without)) if without else None}}

    phases = collections.defaultdict(list)
    for f, k in counts.most_common():
        phases[_classify(f, pres)].append(phase_stats(f, k))
    bal = _balanced_equation(tkey, pres, rows)
    out = {
        "target": tkey,
        "precursors": pres,
        "n_reactions": len(rows),
        "n_papers": n,
        **_paper_states(papers),
        "balanced_equation": bal["equation"],
        "equation_source": bal["source"],
        "phases": dict(phases),
    }
    if n == 0:
        out["note"] = ("No literature for this exact route; see get_literature_recipes(target) "
                       "or find_similar_materials for related evidence.")

    if simulated_phases is not None:
        sim = {_norm_or_raw(f) for f in simulated_phases} - {tkey}
        assess = collections.defaultdict(list)
        for f in sorted(sim):
            cls, k = _classify(f, pres), counts.get(f, 0)
            entry = {"formula": f, "papers": k, "of": n, "ci95": _wilson(k, n)}
            if cls == "secondary phase":
                entry["label"] = ("no literature" if n == 0 else "confirmed" if k
                                  else "rarely reported" if entry["ci95"][1] < RARE_UPPER
                                  else "inconclusive")
                assess["secondary_phases"].append(entry)
            elif cls == "leftover precursor":
                entry["meaning"] = "incomplete conversion"
                assess["leftover_precursors"].append(entry)
            else:
                assess["other"].append({**entry, "class": cls})
        for f, k in counts.most_common():
            if f not in sim and _classify(f, pres) == "secondary phase" and _wilson(k, n)[0] >= MISSED_LOWER:
                assess["missed_by_simulation"].append(
                    {"formula": f, "papers": k, "of": n, "ci95": _wilson(k, n)})
        out["simulation"] = dict(assess)

    if reactant_amounts:
        st = _stoichiometry(tkey, pres, reactant_amounts, bal["left"])
        out["stoichiometry"] = st
        if st.get("status") == "off_stoichiometry" and "simulation" in out:
            out["simulation"]["note"] = (
                "Simulated ratio is off-stoichiometry; literature equations are balanced, so "
                "differences from the literature may come from the ratio.")
    return out


@mcp.tool()
@_quiet
def check_stoichiometry(target: str, reactant_amounts: dict[str, float]) -> dict:
    """Check a simulation recipe's molar amounts (e.g. a ReactCA reactants dict) against
    the balanced equation for the target from those precursors: the dataset's when the
    route is in the literature, else synthesis-similarity's balancer. Returns status
    balanced | off_stoichiometry (limiting element, % excess of the others, missing
    elements), extra elements (flux/dopant), the balanced equation, and for
    off-stoichiometry the balanced amounts. Off-stoichiometry runs are valid data
    points: report them, flagged, and also simulate the balanced amounts."""
    for f in [target, *reactant_amounts]:
        if err := _check_formula(f):
            return {"error": err}
    tkey = _normalize(target)
    rows = _route_rows(tkey, "+".join(sorted({_normalize(p) for p in reactant_amounts}))) if LIT_T else []
    bal = _balanced_equation(tkey, list(reactant_amounts), rows)
    return {"target": tkey, "balanced_equation": bal["equation"], "equation_source": bal["source"],
            **_stoichiometry(tkey, list(reactant_amounts), reactant_amounts, bal["left"])}


@mcp.tool()
@_quiet
def complete_precursor_set(
    target: str, fixed_precursors: list[str], top_k: int = 10
) -> dict:
    """Given a target and precursors that MUST be used (e.g. availability or cost
    constraints), predict the remaining precursors with the masked precursor
    completion model. Completions are route-consistent with the fixed precursors
    (e.g. nitrates suggest nitrates). Scores are model confidence (0-1)."""
    if err := _check_formula(target):
        return {"error": err}
    for p in fixed_precursors:
        if err := _check_formula(p):
            return {"error": err}
    if len(fixed_precursors) > REC.max_mats_num - 1:
        return {"error": f"at most {REC.max_mats_num - 1} fixed precursors supported"}
    zero = np.zeros(len(REC.all_elements), dtype=np.float32)
    cond = [formula_to_array(p, REC.all_elements) for p in fixed_precursors]
    cond += [zero] * (REC.max_mats_num - 1 - len(cond))
    _, pre_str_lists = REC.predict_precursor_callback.predict_precursors(
        REC.framework_model,
        target_compositions=np.array([formula_to_array(target, REC.all_elements)]),
        precursors_conditional=np.array([cond]),
        to_print=False,
    )
    fixed_norm = {_normalize(p) for p in fixed_precursors}
    completions = [
        {"precursor": f, "score": round(float(s), 4)}
        for f, s in pre_str_lists[0]
        if f not in fixed_norm
    ][:top_k]
    return {"target": target, "fixed": fixed_precursors, "completions": completions}


@mcp.tool()
@_quiet
def balance_reaction(target: str, precursors: list[str]) -> dict:
    """Check whether a target can be made from the given precursors as a balanced
    reaction (volatile byproducts like CO2/H2O/O2/NH3 allowed). Returns the balanced
    equation, or balanced=false if stoichiometrically impossible."""
    if err := _check_formula(target):
        return {"error": err}
    for p in precursors:
        if err := _check_formula(p):
            return {"error": err}
    rxn = _balance(target, precursors)
    if rxn is None:
        return {"target": target, "precursors": precursors, "balanced": False}
    return {
        "target": target,
        "precursors": precursors,
        "balanced": True,
        "reaction": rxn[3],
    }


def _selftest():
    r = find_similar_materials("LiFePO4", k=3)
    assert r["neighbors"][0]["similarity"] > 0.99, r
    r = get_literature_recipes("BaTiO3")
    assert r["recipes"] and r["recipes"][0]["sources"][0]["doi"], r
    r = complete_precursor_set("LaAlO3", ["La(NO3)3"], top_k=5)
    assert r["completions"] and all(
        0.0 <= c["score"] <= 1.0 for c in r["completions"]
    ), r
    # elemental completions must be possible (compounds-only blacklist + v0.5
    # vocabulary): this exact case failed under the old all-elements default
    r = complete_precursor_set("K8Na2(FeSe)25", ["Fe", "K", "Se"], top_k=10)
    assert any(c["precursor"] == "Na" for c in r["completions"]), r
    r = balance_reaction("BaTiO3", ["BaCO3", "TiO2"])
    assert r["balanced"] and "BaTiO3" in r["reaction"], r
    r = balance_reaction("BaTiO3", ["BaCO3", "SiO2"])
    assert r.get("balanced") is False, r
    r = recommend_precursors("LiNi0.5Mn1.5O4", top_n=3)
    assert r["recommendations"] and r["recommendations"][0]["reaction"], r
    # first set may be the common-precursors attempt (precedent: None);
    # at least one set must carry an exact literature precedent with DOI
    cited = [
        p["precedent"]
        for p in r["recommendations"]
        if p["precedent"]["reference_material"]
    ]
    assert cited and cited[0]["sources"][0]["doi"], r
    assert "added_beyond_reference" in cited[0] and "reported_set" in cited[0], cited
    r = recommend_precursors("LiFePO4", top_n=3, validate=False)
    assert len(r["recommendations"]) >= 2, r
    assert any(
        p["precedent"]["reference_material"] for p in r["recommendations"]
    ), r
    bad = recommend_precursors("NotAFormula!!")
    assert "error" in bad, bad

    # literature.db keys match the KB exactly; papers are distinct DOIs (<= reactions)
    r = get_literature_recipes("BaTiO3")
    s = next(x for x in r["recipes"] if x["precursors"] == ["BaCO3", "TiO2"])
    assert s["n_reactions"] == 267 and s["n_papers"] == 265, s
    assert any(p["formula"] == "Ba2TiO4" and p["class"] == "secondary phase"
               for p in s["impurities"]["most_reported"]), s
    assert all(x.get("n_papers", 0) <= x["n_reactions"] for x in r["recipes"]), r
    # route_impurities: by-paper counts, phase classes, labels; off-stoichiometry flagged, not dropped
    r = route_impurities("Li4SiO4", ["SiO2", "Li2CO3"],
                         simulated_phases=["Li4SiO4", "Li2SiO3", "Li2CO3"],
                         reactant_amounts={"Li2CO3": 1, "SiO2": 1})
    assert r["n_papers"] == 53 and r["equation_source"] == "dataset", r
    sec = r["simulation"]["secondary_phases"][0]
    assert sec["formula"] == "Li2SiO3" and sec["papers"] == 12 and sec["label"] == "confirmed", r
    assert r["simulation"]["leftover_precursors"][0]["formula"] == "Li2CO3", r
    st = r["stoichiometry"]
    assert st["status"] == "off_stoichiometry" and st["limiting_element"] == "Li", st
    assert st["balanced_reactant_amounts"] == {"Li2CO3": 2.0, "SiO2": 1.0}, st
    r = route_impurities("CaSnO3", ["CaCO3", "SnO2"], simulated_phases=["Ca2SnO4", "CaO"],
                         reactant_amounts={"CaCO3": 1, "SnO2": 1})
    assert r["simulation"]["secondary_phases"][0]["label"] == "inconclusive", r  # 0 of 2 papers
    assert r["simulation"]["leftover_precursors"][0]["formula"] == "CaO", r      # decomposed CaCO3
    assert r["stoichiometry"]["status"] == "balanced", r
    assert _classify("LiAl(SiO3)2", ["Li2CO3", "SiO2"]) == "contamination"
    r = route_impurities("CaSnO3", ["CaO", "SnO2"], simulated_phases=["Ca2SnO4"])
    assert r["n_papers"] == 0 and r["simulation"]["secondary_phases"][0]["label"] == "no literature", r
    assert r["equation_source"] == "synthesis-similarity" and r["balanced_equation"], r
    r = check_stoichiometry("Li4SiO4", {"Li2CO3": 1, "SiO2": 1})
    assert r["status"] == "off_stoichiometry" and r["equation_source"] == "dataset", r
    assert r["balanced_reactant_amounts"] == {"Li2CO3": 2.0, "SiO2": 1.0}, r
    assert check_stoichiometry("LiTaO3", {"Li2CO3": 0.5, "Ta2O5": 0.5})["status"] == "balanced"
    print("SELFTEST OK")
    r = recommend_precursors("LiNi0.5Mn1.5O4", top_n=3)
    for rec_ in r["recommendations"]:
        p = rec_["precedent"]
        print(
            " ",
            rec_["reaction"],
            "<-",
            p["reference_material"],
            f"(added: {p.get('added_beyond_reference')}, doi: {p['sources'][0]['doi'] if p.get('sources') else None})"
            if p["reference_material"]
            else "(common-precursor baseline)",
        )


def main():
    if "--selftest" in sys.argv:
        _selftest()
    else:
        mcp.run()


if __name__ == "__main__":
    main()
