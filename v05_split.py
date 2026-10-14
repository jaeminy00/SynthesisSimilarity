"""Build the v0.5 training split: our converted SS_rxns KB minus benchmark leakage.

Training pool = rsc_impurity/ss_rxns_kb.npz reactions, EXCLUDING any reaction that
  (a) shares a DOI with a val/test reaction of the published split, or
  (b) has a target composition matching any val/test target.
val/test are copied verbatim from the published data_split.npz, so v0.5 numbers
are directly comparable to the paper-protocol table (82.4% test top-5 etc.).

    .venv/bin/python v05_split.py   ->  rsc_impurity/v05_split.npz
"""
import os

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
PUBLIC = os.path.join(REPO, "SynthesisSimilarity/rsc/data_split.npz")
OURS = os.path.join(REPO, "SynthesisSimilarity/rsc_impurity/ss_rxns_kb.npz")
OUT = os.path.join(REPO, "SynthesisSimilarity/rsc_impurity/v05_split.npz")


def norm_doi(d):
    return (d or "").strip().lower()


def comp_key(arr):
    return tuple(np.round(arr, 6))


def main():
    pub = np.load(PUBLIC, allow_pickle=True)
    held_dois, held_comps = set(), set()
    for split in ("val_reactions", "test_reactions"):
        for r in pub[split]:
            held_dois.add(norm_doi(r.get("doi")))
            for c in r["target_comp"]:
                held_comps.add(comp_key(c))
    held_dois.discard("")
    print(f"held-out: {len(held_dois)} DOIs, {len(held_comps)} target compositions")

    # training constraint: model's max_mats_num=6 -> at most 5 precursors
    # (long recipes stay in the v0 KB; they just can't be training samples)
    MAX_PRECURSORS = 5

    ours = np.load(OURS, allow_pickle=True)["train_reactions"]
    kept, cut_doi, cut_comp, cut_long = [], 0, 0, 0
    for r in ours:
        if norm_doi(r.get("doi")) in held_dois:
            cut_doi += 1
            continue
        if comp_key(r["target_comp"][0]) in held_comps:
            cut_comp += 1
            continue
        if len(r["precursors_comp"]) > MAX_PRECURSORS:
            cut_long += 1
            continue
        kept.append(r)

    np.savez(
        OUT,
        train_reactions=np.array(kept, dtype=object),
        val_reactions=pub["val_reactions"],
        test_reactions=pub["test_reactions"],
    )
    print(
        f"train: {len(kept)} of {len(ours)} "
        f"(excluded {cut_doi} by DOI, {cut_comp} by target composition, "
        f"{cut_long} with >{MAX_PRECURSORS} precursors); "
        f"val/test copied from public split"
    )


if __name__ == "__main__":
    main()
