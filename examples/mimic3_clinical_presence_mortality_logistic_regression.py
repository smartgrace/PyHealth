
"""Imputation strategies and group fairness on MIMIC-III (ablation example).

Reproduces the MIMIC-III experiment of Jeanselme et al. (2022), "Imputation
Strategies Under Clinical Presence: Impact on Algorithmic Fairness", ML4H,
PMLR 193: https://proceedings.mlr.press/v193/jeanselme22a.html

Pipeline
    1. ``ClinicalPresenceMortalityMIMIC3`` builds one row per patient: the
       most recent value of each lab in the first 24 h (NaN if never
       measured), death within 8 days, and Black / female / public flags.
    2. ``assign_splits`` splits patients like the authors' code.
    3. ``impute_cohort`` fills the missing labs (fitted on the training pool).
    4. PyHealth's ``LogisticRegression`` is trained once per imputation
       draw (10 for MICE in the paper) and test predictions are averaged.
       scikit-learn's L2 logistic regression with the authors' C grid is run
       as a cross-check.
    5. Metrics as in the paper: AUC per group, and with the top 30% predicted
       risk prioritised, the prioritisation rate and false negative rate
       (FNR) per group. Gaps are group minus the rest; bootstrapped (100x).

Ablations
    A. The paper's four strategies: Median, MICE, Group MICE, Group MICE
       Missing (task configuration ``strategy`` / ``add_missing_indicators``).
    B. Extension: MissForest and Group MissForest (non-linear imputation).
    C. Imputation fitted on ``train`` only instead of the authors' whole
       training pool (removes the small leak from internal-test patients).

Usage
    Synthetic data (no credentials needed, runs in a few minutes):
        python examples/mimic3_clinical_presence_mortality_logistic_regression.py
    Real MIMIC-III v1.4:
        python examples/mimic3_clinical_presence_mortality_logistic_regression.py \\
            --root /path/to/mimiciii/1.4 --draws 10

Results
    (1) synthetic default
    Synthetic data: 2000 patients, 3 MICE draws, 10 epochs, seed 0.

 Configuration                        AUC       skAUC  Black Prio   Black FNR
A. Median                          0.841       0.842       0.032       0.059
A. MICE                            0.855       0.855       0.081       0.005
A. Group MICE                      0.856       0.855       0.081       0.005
A. Group MICE Missing              0.857       0.855       0.033       0.003
B. MissForest                      0.859       0.855       0.064       0.043
B. Group MissForest                0.856       0.855       0.063       0.084
C. MICE, fit on train only         0.855       0.854       0.079       0.058

Configuration                Female Prio  Female FNR Public Prio  Public FNR
A. Median                          0.068       0.066      -0.016       0.017
A. MICE                            0.061       0.084       0.008      -0.049
A. Group MICE                      0.061       0.084       0.008      -0.049
A. Group MICE Missing              0.061       0.084      -0.016      -0.054
B. MissForest                      0.016       0.172       0.024      -0.097
B. Group MissForest                0.024       0.175       0.016      -0.092
C. MICE, fit on train only         0.047       0.148       0.018      -0.117

AUC = PyHealth LogisticRegression; skAUC = scikit-learn cross-check.
Gaps = group minus rest on the test set, bootstrap mean (100x).

    (2) the real MIMIC-III

"""

import argparse
import gzip
import tempfile
import time 
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from pyhealth.datasets import MIMIC3Dataset, create_sample_dataset, get_dataloader
from pyhealth.models import LogisticRegression
from pyhealth.tasks.clinical_presence_mortality_mimic3 import (
    PAPER_LAB_GROUPS,
    ClinicalPresenceMortalityMIMIC3,
    assign_splits,
    impute_cohort,
)
from pyhealth.trainer import Trainer

GROUPS = ["black", "female", "public"]
TOP_FRACTION = 0.3  # the paper's policy: priority care for the top 30% risk


# ----------------------------------------------------------------------
# 1. Synthetic MIMIC-III-like data with clinical presence built in
# ----------------------------------------------------------------------
def _write_gz(path, header, rows):
    with gzip.open(path, "wt") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")


def make_synthetic_mimic3(folder, n_patients=2000, n_labs=10, seed=0):
    """Write fake PATIENTS/ADMISSIONS/ICUSTAYS/D_LABITEMS/LABEVENTS files.

    Missingness follows the paper's scenarios: sicker patients are tested
    more (S3, confirmation bias), and Black or publicly insured patients are
    tested less at the same severity (S1, access to care).
    """
    rng = np.random.default_rng(seed)
    folder = Path(folder)
    labs = list(PAPER_LAB_GROUPS.items())[:n_labs]
    patients, admissions, labevents = [], [], []
    for pid in range(1, n_patients + 1):
        black, female, public = (rng.random(3) < [0.15, 0.45, 0.6]).astype(int)
        severity = rng.normal()
        died = rng.random() < 1 / (1 + np.exp(-(severity * 1.5 - 2.2)))
        admit = np.datetime64("2150-01-01T08:00") + np.timedelta64(pid, "D")
        stay_days = 3 if died else 6
        discharge = admit + np.timedelta64(stay_days, "D")
        patients.append([pid, pid, "F" if female else "M", "2090-01-01 00:00:00",
                         "", "", "", int(died)])
        admissions.append([
            pid, pid, 1000 + pid, str(admit).replace("T", " ") + ":00",
            str(discharge).replace("T", " ") + ":00",
            str(discharge).replace("T", " ") + ":00" if died else "",
            "EMERGENCY", "", "", "Medicare" if public else "Private", "", "",
            "", "BLACK/AFRICAN AMERICAN" if black else "WHITE", "", "", "",
            int(died), 1,
        ])
        p_test = 1 / (1 + np.exp(-(0.3 + 0.8 * severity - 0.9 * black
                                   - 0.6 * public)))
        for j, (_, itemids) in enumerate(labs):
            if rng.random() > p_test:
                continue  # this lab was never ordered
            value = 10 + j + severity * (1 + 0.3 * j) + rng.normal()
            hours = rng.uniform(0.5, 23.5)
            time = admit + np.timedelta64(int(hours * 60), "m")
            labevents.append([len(labevents) + 1, pid, 1000 + pid, itemids[0],
                              str(time).replace("T", " ") + ":00", value,
                              round(value, 3), "", ""])
    _write_gz(folder / "PATIENTS.csv.gz",
              ["row_id", "subject_id", "gender", "dob", "dod", "dod_hosp",
               "dod_ssn", "expire_flag"], patients)
    _write_gz(folder / "ADMISSIONS.csv.gz",
              ["row_id", "subject_id", "hadm_id", "admittime", "dischtime",
               "deathtime", "admission_type", "admission_location",
               "discharge_location", "insurance", "language", "religion",
               "marital_status", "ethnicity", "edregtime", "edouttime",
               "diagnosis", "hospital_expire_flag", "has_chartevents_data"],
              admissions)
    _write_gz(folder / "ICUSTAYS.csv.gz",
              ["subject_id", "intime", "icustay_id", "first_careunit",
               "dbsource", "last_careunit", "outtime"], [])
    _write_gz(folder / "D_LABITEMS.csv.gz",
              ["row_id", "itemid", "label", "fluid", "category", "loinc_code"],
              [[i + 1, itemids[0], name.replace(",", " "), "Blood", "Lab", ""]
               for i, (name, itemids) in enumerate(labs)])
    _write_gz(folder / "LABEVENTS.csv.gz",
              ["row_id", "subject_id", "hadm_id", "itemid", "charttime",
               "value", "valuenum", "valueuom", "flag"], labevents)
    return [name for name, _ in labs]


# ----------------------------------------------------------------------
# 2. Models
# ----------------------------------------------------------------------
def standardize(x, fit_rows):
    """Scale each column with the mean / sd of ``fit_rows`` (authors' code)."""
    mean, sd = x[fit_rows].mean(axis=0), x[fit_rows].std(axis=0)
    return (x - mean) / np.where(sd == 0, 1.0, sd)


def pyhealth_predict(task, x, y, train, val, test, epochs, seed):
    """Train PyHealth's LogisticRegression; return test probabilities."""
    def make(rows):
        samples = [{"patient_id": str(i), "features": x[i].tolist(),
                    "mortality": int(y[i])} for i in np.flatnonzero(rows)]
        return create_sample_dataset(samples, task.input_schema,
                                     task.output_schema, task_name=task.task_name)

    import torch

    torch.manual_seed(seed)
    train_ds, val_ds, test_ds = make(train), make(val), make(test)
    model = LogisticRegression(dataset=train_ds)
    trainer = Trainer(model=model, enable_logging=False)
    trainer.train(
        train_dataloader=get_dataloader(train_ds, batch_size=64, shuffle=True),
        val_dataloader=get_dataloader(val_ds, batch_size=256, shuffle=False),
        epochs=epochs, monitor="roc_auc",
    )
    _, y_prob, _ = trainer.inference(
        get_dataloader(test_ds, batch_size=256, shuffle=False)
    )
    return np.asarray(y_prob).reshape(-1)


def sklearn_predict(x, y, train, val, test):
    """Authors' model: L2 logistic regression, C chosen on validation loss."""
    best, best_loss = None, np.inf
    for c in [0.01, 0.1, 1.0, 10.0]:  # authors' grid (lambda = 1 / C)
        model = SkLogisticRegression(C=c, max_iter=1000).fit(x[train], y[train])
        loss = log_loss(y[val], model.predict_proba(x[val])[:, 1], labels=[0, 1])
        if loss < best_loss:
            best, best_loss = model, loss
    return best.predict_proba(x[test])[:, 1]


# ----------------------------------------------------------------------
# 3. The paper's metrics
# ----------------------------------------------------------------------
def group_metrics(y, prob, flags, n_boot=100, seed=0):
    """AUC, prioritisation and FNR gaps (group minus rest), bootstrapped."""
    threshold = np.sort(prob)[::-1][int(len(prob) * TOP_FRACTION) - 1]
    prioritised = prob >= threshold
    rng = np.random.default_rng(seed)
    out = {"AUC": roc_auc_score(y, prob)}
    for name, member in flags.items():
        gaps = {"AUC": [], "Prio": [], "FNR": []}
        for _ in range(n_boot):
            stats = {}
            for side in (member == 1, member == 0):
                idx = rng.choice(np.flatnonzero(side), side.sum(), replace=True)
                yy, pp, pr = y[idx], prob[idx], prioritised[idx]
                auc = roc_auc_score(yy, pp) if 0 < yy.sum() < len(yy) else np.nan
                fnr = (~pr[yy == 1]).mean() if yy.sum() else np.nan
                stats[bool(side[0] == (member[0] == 1))] = (auc, pr.mean(), fnr)
            group, rest = stats[True], stats[False]
            for k, key in enumerate(["AUC", "Prio", "FNR"]):
                gaps[key].append(group[k] - rest[k])
        for key, values in gaps.items():
            out[f"{name} {key} gap"] = np.nanmean(values)
    return out


# ----------------------------------------------------------------------
# 4. Experiment
# ----------------------------------------------------------------------
CONFIGS = {
    "A. Median": {"strategy": "median"},
    "A. MICE": {"strategy": "mice"},
    "A. Group MICE": {"strategy": "group_mice"},
    "A. Group MICE Missing": {"strategy": "group_mice",
                              "add_missing_indicators": True},
    "B. MissForest": {"strategy": "missforest"},
    "B. Group MissForest": {"strategy": "group_missforest"},
    "C. MICE, imputation fit on train only": {"strategy": "mice",
                                              "fit_on": "train"},
}


RANDOM_STRATEGIES = {"mice", "group_mice"}  # one draw is enough for the others


def run(root, lab_names, draws, epochs, seed):
    task = ClinicalPresenceMortalityMIMIC3(lab_names=lab_names)
    dataset = MIMIC3Dataset(root=root, tables=["labevents"])
    raw = list(dataset.set_task(task, num_workers=1))
    ids = [s["patient_id"] for s in raw]
    x_raw = np.stack([np.asarray(s["features"], dtype=float) for s in raw])
    y = np.array([int(s["mortality"]) for s in raw])
    groups = np.array([[s[g] for g in GROUPS] for s in raw])
    split = np.array([assign_splits(ids, seed=seed)[i] for i in ids])
    train, val = split == "train", split == "val"
    test = np.isin(split, ["internal_test", "external_test"])
    pool = split != "external_test"
    flags = {g: groups[test, k] for k, g in enumerate(GROUPS)}
    print(f"{len(ids)} patients, {int(y.sum())} deaths, "
          f"{np.isnan(x_raw).mean():.0%} of lab cells missing")

    rows = {}
    for name, cfg in CONFIGS.items():
        cfg = dict(cfg)
        fit_rows = train if cfg.pop("fit_on", "pool") == "train" else pool
        n_draws = draws if cfg["strategy"] in RANDOM_STRATEGIES else 1
        ph, sk = [], []
        for d in range(n_draws):
            filled = impute_cohort(x_raw, fit_rows, groups=groups,
                                   seed=seed + d, **cfg)
            x = standardize(filled, train)
            ph.append(pyhealth_predict(task, x, y, train, val, test, epochs,
                                       seed + d))
            sk.append(sklearn_predict(x, y, train, val, test))
        rows[name] = (group_metrics(y[test], np.mean(ph, axis=0), flags),
                      group_metrics(y[test], np.mean(sk, axis=0), flags))
        print(f"done: {name} ({n_draws} draw(s))")
    return rows


def print_table(rows):
    """Print two narrow tables (they fit in the 88-character docstring limit)."""
    parts = [
        [("AUC", "AUC", 0), ("skAUC", "AUC", 1), ("Black Prio", "black Prio gap", 0),
         ("Black FNR", "black FNR gap", 0)],
        [("Female Prio", "female Prio gap", 0), ("Female FNR", "female FNR gap", 0),
         ("Public Prio", "public Prio gap", 0), ("Public FNR", "public FNR gap", 0)],
    ]
    for columns in parts:
        print("\n" + f"{'Configuration':28}" + "".join(f"{c:>12}" for c, _, _ in columns))
        for name, results in rows.items():
            name = name.replace("imputation fit", "fit")
            values = "".join(f"{results[i][key]:12.3f}" for _, key, i in columns)
            print(f"{name:28}{values}")
    print("\nAUC = PyHealth LogisticRegression; skAUC = scikit-learn cross-check.")
    print("Gaps = group minus rest on the test set, bootstrap mean (100x).")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", help="real MIMIC-III folder (default: synthetic)")
    parser.add_argument("--patients", type=int, default=2000)
    parser.add_argument("--draws", type=int, default=3, help="MICE draws (paper: 10)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.root:
        print_table(run(args.root, None, args.draws, args.epochs, args.seed))
        return
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        lab_names = make_synthetic_mimic3(folder, args.patients, seed=args.seed)
        print_table(run(folder, lab_names, args.draws, args.epochs, args.seed))


if __name__ == "__main__":
    main()