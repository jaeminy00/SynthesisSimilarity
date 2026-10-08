"""Literature-based precursor recommender MCP server.

Architecture: PrecursorSelector (He et al. Sci. Adv. 2023).
Weights: v0.5 — retrained on the Lee text-mined dataset (66,154 solid-state
reactions, 46,816 papers); benchmark parity with the published model.
Knowledge base: the full converted Lee dataset (28,844 unique targets).
Unavailable-precursor default: compounds-only list (elemental precursors allowed).

Run:      .venv/bin/python mcp_server.py
Selftest: .venv/bin/python mcp_server.py --selftest
"""
import contextlib
import functools
import os
import sys

import numpy as np
from mcp.server import MCPServer

from SynthesisSimilarity import PrecursorsRecommendation
from SynthesisSimilarity.core.utils import formula_to_array, array_to_formula
from SynthesisSimilarity.core.mat_featurization import featurize_list_of_composition

mcp = MCPServer("synthesis-similarity")

REPO = os.path.dirname(os.path.abspath(__file__))

print("loading v0.5 model + 66k-reaction knowledge base (~25s)...", file=sys.stderr)
# The library and Keras print to stdout while loading, but stdout is the JSONRPC
# wire — stray lines there break the client's handshake. Park stdout on stderr
# until the model is up; once serving, the SDK diverts fd 1 to stderr itself.
with contextlib.redirect_stdout(sys.stderr):
    REC = PrecursorsRecommendation(
        model_dir=os.path.join(REPO, "generated/v05"),
        data_path=os.path.join(REPO, "SynthesisSimilarity/rsc_impurity/ss_rxns_kb.npz"),
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


def _max_heating_T(raw_index):
    r = RAW.get(raw_index)
    temps = [
        t["max"]
        for op in (r.get("operations") or [])
        for t in op.get("attributes", {}).get("temperature", [])
        if "Heating" in op["type"] and t.get("max") is not None
    ] if r else []
    return max(temps) if temps else None


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
    material (exact composition match), with paper counts, DOIs, publication years,
    and max heating temperatures. Use find_similar_materials first if the target
    might not be in the knowledge base."""
    if err := _check_formula(target):
        return {"error": err}
    key = _normalize(target)
    recipe = REC.train_targets.get(key)
    if recipe is None:
        return {
            "error": f"'{target}' (normalized: {key}) not in knowledge base; "
            "use find_similar_materials for nearby known targets"
        }
    recipes = [
        {
            "precursors": list(pres_set),
            "n_papers": count,
            "sources": _recipe_sources(recipe, pres_set),
        }
        for pres_set, count in recipe["pres"].most_common()
    ]
    return {
        "target": key,
        "n_reactions": len(recipe["raw_index"]),
        "synthesis_types": dict(recipe["syn_type"]),
        "recipes": recipes,
    }


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
    try:
        rxn = REC.get_reaction(
            target_formula=target, precursors_formulas=precursors, ref_materials_comp={}
        )
    except Exception as e:
        return {"error": f"balancing failed: {e}"}
    if rxn is None:
        from SynthesisSimilarity.scripts_utils.reaction_utils import balance_w_rxn_network

        rxn = balance_w_rxn_network(
            target_formula=target, precursors_formulas=precursors, ref_materials_comp={}
        )
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


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        mcp.run()
