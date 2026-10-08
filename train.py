"""Training driver for the PrecursorSelector multi-task model.

Reconstructs the training run recorded in
models/SynthesisRecommendation/{cmd_parameters.json,model_meta.pkl}:
same model config (incl. data-derived vocabularies), same data split,
same sampling scheme (train_data_generator), same optimizer/schedule.

Replication run (original data):
    .venv/bin/python train.py --out generated/replication
Smoke test:
    .venv/bin/python train.py --out generated/smoke --epochs 2 --steps 50

The output dir is loadable by model_utils.load_framework_model() and thus
usable directly by PrecursorsRecommendation(model_dir=...).
"""
import argparse
import json
import os
import pickle
import random
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras

from SynthesisSimilarity.core import model_framework
from SynthesisSimilarity.scripts_utils.train_utils import (
    train_data_generator,
    prepare_dataset,
    get_mat_dico,
    get_syn_type_dico,
    get_ele_counts,
    get_num_reactions,
)

REPO = os.path.dirname(os.path.abspath(__file__))
SHIPPED_MODEL_DIR = os.path.join(REPO, "SynthesisSimilarity/models/SynthesisRecommendation")
DATA_PATH = os.path.join(REPO, "SynthesisSimilarity/rsc/data_split.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output model dir")
    ap.add_argument(
        "--data",
        default=DATA_PATH,
        help="split npz (train/val/test_reactions). If not the published split, "
        "the data-derived vocabularies (precursor dico, syn types, element "
        "counts) are rebuilt from its train reactions, as train_opt.py did.",
    )
    # defaults = cmd_parameters.json of the released model
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--steps", type=int, default=10000, help="steps per epoch")
    ap.add_argument("--val-pairs", type=int, default=8192)
    ap.add_argument("--seed", default="Similarity", help="random_seed_str of original run")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(len(args.seed))
    tf.random.set_seed(len(args.seed))

    # ---- config: reuse the released run's full config (incl. vocabularies) ----
    with open(os.path.join(SHIPPED_MODEL_DIR, "model_meta.pkl"), "rb") as fr:
        model_meta = pickle.load(fr)
    config = dict(model_meta["config"])
    out_dir = os.path.abspath(args.out)
    os.makedirs(os.path.join(out_dir, "saved_model"), exist_ok=True)
    config["model_path"] = out_dir
    config["num_train_steps"] = args.epochs * args.steps  # only used by adamdecay

    # ---- data: train for sampling, val for selection ----
    data = np.load(args.data, allow_pickle=True)
    train_reactions = list(data["train_reactions"])
    val_reactions = list(data["val_reactions"])
    print(
        f"reactions: train={len(train_reactions)} val={len(val_reactions)} "
        f"(config num_train_reactions={config['num_train_reactions']})"
    )

    if os.path.abspath(args.data) != os.path.abspath(DATA_PATH):
        # custom dataset: rebuild data-derived vocabularies exactly as the
        # original driver (train_opt.py) did; keep all hyperparameters and the
        # element basis (all_eles) from the released config
        n_res = config["num_reserved_ids"]
        mat_labels, mat_compositions, mat_counts = get_mat_dico(
            train_reactions, mode="precursor", num_reserved_ids=n_res
        )
        syn_type_labels, syn_type_counts = get_syn_type_dico(
            train_reactions, num_reserved_ids=n_res
        )
        config.update(
            mat_labels=mat_labels,
            mat_compositions=mat_compositions,
            mat_counts=mat_counts,
            syn_type_labels=syn_type_labels,
            syn_type_counts=syn_type_counts,
            ele_counts=get_ele_counts(train_reactions),
            num_train_reactions=get_num_reactions(train_reactions),
        )
        print(
            f"rebuilt vocab: {len(mat_labels)} precursor labels, "
            f"{len(syn_type_labels)} syn types, "
            f"num_train_reactions={config['num_train_reactions']}"
        )

    batch_size = config["batch_size"]
    train_x, train_y = train_data_generator(
        train_reactions,
        num_batch=args.steps,
        max_mats_num=config["max_mats_num"],
        batch_size=batch_size,
        # cmd_parameters.json: precursor_drop_n = -5
        precursor_drop_n=-5,
    )
    train_ds = tf.data.Dataset.zip((train_x, train_y)).prefetch(tf.data.AUTOTUNE)

    # static val set: pair sampling over val reactions, fixed seed
    n_val = len(val_reactions)
    val_ratio = min(1.0, 2 * args.val_pairs / (n_val * (n_val - 1)))
    val_x, val_y = prepare_dataset(
        val_reactions,
        max_mats_num=config["max_mats_num"],
        batch_size=batch_size,
        sampling_ratio=val_ratio,
        precursor_drop_n=-5,
        random_seed=42,
    )
    val_ds = tf.data.Dataset.zip((val_x, val_y)).cache()

    # ---- model: identical construction to load_framework_model, no weight load ----
    model = model_framework.MultiTasksOnRecipes(**config)

    cp_path = os.path.join(out_dir, "saved_model", "cp.ckpt")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            cp_path,
            save_weights_only=True,
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1,
        ),
        keras.callbacks.CSVLogger(os.path.join(out_dir, "training_log.csv")),
        keras.callbacks.TerminateOnNaN(),
    ]

    # save meta first so the dir is loadable as soon as a checkpoint exists
    with open(os.path.join(out_dir, "model_meta.pkl"), "wb") as fw:
        pickle.dump({"config": config}, fw)
    with open(os.path.join(out_dir, "train_args.json"), "w") as fw:
        json.dump(vars(args), fw, indent=2)

    model.fit(
        train_ds,
        epochs=args.epochs,
        steps_per_epoch=args.steps,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=2,
    )
    print("done; best checkpoint at", cp_path)


if __name__ == "__main__":
    main()
