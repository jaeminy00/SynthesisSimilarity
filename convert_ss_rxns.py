"""Convert the SS_rxns impurity dataset to the repo's knowledge-base format.

Input:  text-mined reactions JSON (target/precursors material dicts,
        impurity_phase, conditions_forDOI, DOI).
Output: rsc_impurity/ss_rxns_kb.npz          — loadable as PrecursorsRecommendation
                                               data_path (all reactions in
                                               train_reactions; val/test empty)
        rsc_impurity/impurity_annotations.json — {raw_index: [impurity formulas]}
                                               for every kept reaction reporting
                                               impurities; empty list = impure
                                               with unidentified phase(s)

    .venv/bin/python convert_ss_rxns.py --input PATH/SS_rxns_80806.json
"""
import argparse
import json
import os
import pickle
from collections import Counter

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(REPO, "SynthesisSimilarity/rsc_impurity")
META = os.path.join(REPO, "SynthesisSimilarity/models/SynthesisRecommendation/model_meta.pkl")


def material_to_comp(mat, ele_index):
    """Material dict -> normalized composition array (fragments summed by amount).

    Returns None if any amount is non-numeric, any element is outside the
    model vocabulary, or the composition is empty.
    """
    totals = Counter()
    for frag in mat.get("composition") or []:
        try:
            frag_amount = float(frag.get("amount", 1))
        except (TypeError, ValueError):
            return None
        for ele, amt in (frag.get("elements") or {}).items():
            if ele not in ele_index:
                return None
            try:
                totals[ele] += float(amt) * frag_amount
            except (TypeError, ValueError):
                return None
    if not totals or min(totals.values()) < 0:
        return None
    arr = np.zeros(len(ele_index), dtype=np.float32)
    for ele, amt in totals.items():
        arr[ele_index[ele]] = amt
    s = arr.sum()
    if s <= 0:
        return None
    return arr / s


def convert_ops(conditions):
    """conditions_forDOI acts -> npz-style operations (for max-heating-T lookups)."""
    ops = []
    for act in conditions or []:
        ops.append(
            {
                "type": (act.get("act_type") or "") + "Operation",
                "attributes": {"temperature": act.get("temp_values") or []},
            }
        )
    return ops


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True,
                    help="SS_rxns_80806.json (Lee et al., Sci. Data 2025)")
    args = ap.parse_args()

    with open(META, "rb") as fr:
        all_eles = pickle.load(fr)["config"]["all_eles"]
    ele_index = {e: i for i, e in enumerate(all_eles)}

    print(f"loading {args.input}")
    with open(args.input) as fr:
        data = json.load(fr)

    reactions = []
    impurities = {}
    skipped = Counter()
    for i, r in enumerate(data):
        tar_mats = r.get("target") or []
        pre_mats = r.get("precursors") or []
        if not any(m.get("material_formula") for m in tar_mats):
            skipped["no_target_formula"] += 1
            continue
        if any(m.get("elements_vars") for m in tar_mats + pre_mats):
            skipped["elements_vars"] += 1
            continue
        # dataset has no multi-target records; guard anyway
        if len(tar_mats) != 1:
            skipped["multi_target"] += 1
            continue

        tar_comp = material_to_comp(tar_mats[0], ele_index)
        pre_comps = [material_to_comp(m, ele_index) for m in pre_mats]
        if tar_comp is None or not pre_comps or any(p is None for p in pre_comps):
            skipped["unparseable_composition"] += 1
            continue
        # KB code requires unique precursors per reaction; entries with the
        # same normalized composition are the same material to this model
        seen, unique_pres = set(), []
        for p in pre_comps:
            key = tuple(np.round(p, 6))
            if key not in seen:
                seen.add(key)
                unique_pres.append(p)
        pre_comps = unique_pres

        reactions.append(
            {
                "raw_index": i,
                "id": i,
                "doi": r.get("DOI"),
                "count_weight": 1.0,
                "synthesis_type": "solid_state_ceramic_synthesis",
                "operations": convert_ops(r.get("conditions_forDOI")),
                # identity featurizer (featurizer_type='default'):
                # featurized comps == normalized composition arrays
                "target_comp": [tar_comp],
                "target_comp_featurized": [tar_comp],
                "precursors_comp": [[p] for p in pre_comps],
                "precursors_comp_featurized": [[p] for p in pre_comps],
            }
        )

        # any impurity_phase report marks the reaction phase-impure;
        # 'unknown' phases keep the impure signal with an empty formula list
        if r.get("impurity_phase"):
            impurities[i] = [
                m["material_formula"]
                for m in r["impurity_phase"]
                if m.get("material_formula") and m["material_formula"] != "unknown"
            ]

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(
        os.path.join(OUT_DIR, "ss_rxns_kb.npz"),
        train_reactions=np.array(reactions, dtype=object),
        val_reactions=np.array([], dtype=object),
        test_reactions=np.array([], dtype=object),
    )
    with open(os.path.join(OUT_DIR, "impurity_annotations.json"), "w") as fw:
        json.dump(impurities, fw)

    n_imp = sum(1 for x in reactions if x["raw_index"] in impurities)
    print(f"kept {len(reactions)} / {len(data)} reactions; skipped: {dict(skipped)}")
    print(f"impurity-annotated kept reactions: {n_imp} (sidecar entries: {len(impurities)})")


if __name__ == "__main__":
    main()
