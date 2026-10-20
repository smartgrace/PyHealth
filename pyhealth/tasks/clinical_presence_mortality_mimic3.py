"""Clinical-presence mortality task for MIMIC-III."""

import math
import warnings
from collections import Counter
from datetime import datetime, timedelta
from typing import Any, ClassVar

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression

from .base_task import BaseTask


def _to_number(value: Any) -> float | None:
    """Return ``value`` as a float, or None if it is blank or not a number."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _parse_time(value: str) -> datetime:
    """Parse a MIMIC-III date-time string.

    MIMIC-III times are date-shifted local times with no time zone, and
    PyHealth's event timestamps are naive too, so the result stays naive.
    """
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")  # noqa: DTZ007



#: The 67 laboratory variables of the paper, as grouped by MIMIC-Extract
#: (Wang et al., 2020; ``resources/itemid_to_variable_map.csv``, rows with
#: ``STATUS == 'ready'``, ``COUNT > 0`` and ``LINKSTO == 'labevents'``, which
#: is the selection of the authors' ``preprocessing.ipynb``). Each variable
#: (``LEVEL1`` name) pools one or more MIMIC-III ``LABEVENTS`` ``itemid``
#: values; fluids are kept apart by name, e.g. ``"Albumin"`` vs
#: ``"Albumin (ascites)"``.
PAPER_LAB_GROUPS: dict[str, tuple[int, ...]] = {
    "Alanine aminotransferase": (50861,),
    "Albumin": (50862,),
    "Albumin (ascites)": (50835,),
    "Albumin (pleural)": (51046,),
    "Albumin (urine)": (51069,),
    "Alkaline phosphate": (50863,),
    "Anion gap": (50868,),
    "Asparate aminotransferase": (50878,),
    "Basophils": (51146,),
    "Bicarbonate": (50882,),
    "Bicarbonate (whole blood)": (50803,),
    "Bilirubin (conjugated)": (50883,),
    "Bilirubin (total)": (50885,),
    "Bilirubin (unconjugated)": (50884,),
    "Blood urea nitrogen": (51006,),
    "Calcium (total)": (50893,),
    "Calcium (urine)": (51077,),
    "Calcium ionized": (50808,),
    "Chloride": (50902,),
    "Chloride (urine)": (51078,),
    "Chloride (whole blood)": (50806,),
    "Cholesterol (HDL)": (50904,),
    "Cholesterol (LDL)": (50905,),
    "Cholesterol (total)": (50907,),
    "CO2 (ETCO2, PCO2, etc.)": (50804,),
    "Creatinine": (50912,),
    "Creatinine (ascites)": (50841,),
    "Creatinine (body fluid)": (51032,),
    "Creatinine (pleural)": (51052,),
    "Creatinine (urine)": (51082,),
    "Eosinophils": (51114, 51347, 51419, 51444),
    "Fibrinogen": (51214,),
    "Glucose": (50809, 50931),
    "Hematocrit": (50810, 51221),
    "Hemoglobin": (50811, 51222),
    "Lactate": (50813,),
    "Lactate dehydrogenase": (50954, 51054),
    "Lymphocytes": (51116, 51244, 51375, 51427, 51446),
    "Lymphocytes (atypical)": (51143, 51343),
    "Lymphocytes (percent)": (51245,),
    "Magnesium": (50960,),
    "Mean corpuscular hemoglobin": (51248,),
    "Mean corpuscular hemoglobin concentration": (51249,),
    "Mean corpuscular volume": (51250,),
    "Monocytes": (51254, 51355),
    "Neutrophils": (51256,),
    "Oxygen saturation": (50817,),
    "Partial pressure of carbon dioxide": (50818,),
    "Partial thromboplastin time": (51275,),
    "pH": (50820, 50831, 51094, 51491),
    "Phosphate": (50970,),
    "Platelets": (51265,),
    "Positive end-expiratory pressure": (50819,),
    "Potassium": (50971,),
    "Potassium (whole blood)": (50822,),
    "Prothrombin time": (51237, 51274),
    "Red blood cell count": (51279, 51493),
    "Red blood cell count (ascites)": (51127,),
    "Red blood cell count (CSF)": (51362,),
    "Red blood cell count (pleural)": (51457,),
    "Sodium": (50983,),
    "Sodium (whole blood)": (50824,),
    "Tidal Volume Observed": (50826,),
    "Total Protein Urine": (51102,),
    "Troponin-I": (51002,),
    "Troponin-T": (51003,),
    "White blood cell count": (51300, 51301, 51516),
}



#: Imputation strategies understood by :func:`impute_cohort`.
STRATEGIES = ("median", "mice", "group_mice", "missforest", "group_missforest")


def impute_cohort(
    features: np.ndarray,
    train_mask: np.ndarray,
    strategy: str = "median",
    groups: np.ndarray | None = None,
    add_missing_indicators: bool = False,
    max_iter: int = 10,
    seed: int = 0,
) -> np.ndarray:
    """Fill the missing labs of a whole cohort, learning only from training rows.

    Follows the paper's code (Jeanselme et al., 2022, commit fcb580b):

    * ``"median"``: each missing lab gets that lab's median over training rows.
    * ``"mice"``: start from the median fill, then, ``max_iter`` times, go
      through the labs with missing values (fewest missing first). For each
      lab, fit a linear regression on training rows where the lab was
      measured, using every other column as predictors, and replace the
      missing values by the prediction plus random noise.
    * ``"group_mice"``: MICE with the ``groups`` columns as extra predictors;
      they are removed from the output.
    * ``"missforest"`` / ``"group_missforest"`` (our extension, not in the
      paper): like (Group) MICE, but each lab is predicted by a forest of
      decision trees instead of a straight line (scikit-learn's
      MissForest-style ``IterativeImputer`` with ``ExtraTreesRegressor``).  

    With ``add_missing_indicators=True``, one 0/1 column per lab
    (1 = was missing) is appended. As in the authors' code, it is added
    before imputing, so MICE also uses it as a predictor.

    Args:
        features: ``(patients, labs)`` array; ``NaN`` means missing.
        train_mask: ``(patients,)`` True for training rows.
        strategy: ``"median"``, ``"mice"`` or ``"group_mice"``.
        groups: ``(patients, g)`` 0/1 group flags, needed for ``"group_mice"``.
        add_missing_indicators: Append the missingness indicators.
        max_iter: Passes over the labs for MICE (paper: 10).
        seed: Seed for the MICE noise; a different seed gives a different draw.

    Returns:
        ``(patients, labs)`` array without ``NaN``, or ``(patients, 2 * labs)``
        with the indicators.

    Examples:
        >>> x = np.array([[1.0, np.nan], [3.0, 4.0], [np.nan, 6.0]])
        >>> train = np.array([True, True, True])
        >>> impute_cohort(x, train, strategy="median")
        array([[1., 5.],
               [3., 4.],
               [2., 6.]])
    """


    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    use_groups = strategy.startswith("group_")
    if use_groups and groups is None:
        raise ValueError(f"{strategy} needs the groups array") 
    

    labs = np.asarray(features, dtype=float)
    missing = np.isnan(labs)
    n_labs = labs.shape[1]

    # Columns: labs, then (optional) missing indicators, then (optional) groups
    blocks = [labs]

    if add_missing_indicators:
        blocks.append(missing.astype(float))

    if use_groups:
        blocks.append(np.asarray(groups, dtype=float).reshape(len(labs), -1))
    work = np.hstack(blocks)
    n_keep = work.shape[1] - (blocks[-1].shape[1] if use_groups else 0)

    if strategy.endswith("missforest"):
        # Extension: tree-based chained imputation, fitted on training rows
        forest = IterativeImputer(
            estimator=ExtraTreesRegressor(
                n_estimators=50, min_samples_leaf=3, random_state=seed
            ),
            initial_strategy="median",
            max_iter=max_iter,
            keep_empty_features=True,
            random_state=seed,
        )
        with warnings.catch_warnings():  # fixed rounds, not a convergence error
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            forest.fit(work[train_mask])
            work = forest.transform(work)
        return work[:, :n_keep]


    # Step 1: median of each column over TRAINING rows only
    # (a lab never measured in training gets 0)
    medians = np.nan_to_num(np.nanmedian(work[train_mask], axis=0), nan=0.0)
    for col in range(n_labs):
        work[missing[:, col], col] = medians[col]
    if strategy == "median":
        return work[:, :n_keep]

    # Step 2 (MICE): labs with missing values, fewest missing first
    order = [c for c in np.argsort(missing.sum(axis=0), kind="stable")
             if missing[:, c].any()]
    rng = np.random.default_rng(seed)
    for _ in range(max_iter):
        for col in order:
            others = [c for c in range(work.shape[1]) if c != col]
            fit_rows = train_mask & ~missing[:, col]
            if fit_rows.sum() < 2:
                continue  # too few measured values to learn from
            model = LinearRegression().fit(
                work[fit_rows][:, others], work[fit_rows, col]
            )
            # Authors: noise sd = std of the ABSOLUTE training residuals
            residuals = np.abs(
                model.predict(work[fit_rows][:, others]) - work[fit_rows, col]
            )
            rows = missing[:, col]
            work[rows, col] = model.predict(work[rows][:, others]) + rng.normal(
                0.0, residuals.std(), size=rows.sum()
            )
    return work[:, :n_keep]




def assign_splits(patient_ids: list[str], seed: int = 0) -> dict[str, str]:
    """Split patients the way the authors' code does (commit fcb580b).

    * 20% of patients: ``"external_test"``.
    * The other 80% (the "training pool") is what imputation learns from.
      Inside it: 10% ``"internal_test"``, then 90% / 10% of the rest for
      ``"train"`` (fits the model) and ``"val"`` (picks its settings).

    The authors report results on ``internal_test`` + ``external_test``.

    Args:
        patient_ids: One id per patient.
        seed: Random seed, so the same seed always gives the same split.

    Returns:
        Mapping from patient id to its split name.

    Examples:
        >>> splits = assign_splits([str(i) for i in range(100)])
        >>> sorted(Counter(splits.values()).items())
        [('external_test', 20), ('internal_test', 8), ('train', 65), ('val', 7)]
    """
    ids = sorted(set(patient_ids))
    order = [ids[i] for i in np.random.default_rng(seed).permutation(len(ids))]
    n_pool = round(0.8 * len(ids))
    pool, external = order[:n_pool], order[n_pool:]
    n_internal = round(0.1 * len(pool))
    internal, rest = pool[:n_internal], pool[n_internal:]
    n_val = round(0.1 * len(rest))
    val, train = rest[:n_val], rest[n_val:]
    splits = {}
    for name, members in [
        ("train", train),
        ("val", val),
        ("internal_test", internal),
        ("external_test", external),
    ]:
        for patient_id in members:
            splits[patient_id] = name
    return splits




class ClinicalPresenceMortalityMIMIC3(BaseTask):
    """One sample per patient (last admission): first-day labs, label, groups.

    Reproduces the MIMIC-III cohort of Jeanselme et al. (2022), following the
    authors' ``preprocessing.ipynb``: each adult patient's last hospital
    admission lasting at least 1 day, the most recent value of each lab
    variable strictly inside the first 24 hours (``NaN`` if not measured),
    and a label for death less than 8 days after admission. Patients with no
    lab in the window are dropped, and a lab recorded twice at the same time
    is dropped entirely. Missing labs are left as ``NaN`` so that
    :func:`impute_cohort` can be fitted on training patients only.

    Each sample has ``patient_id``, ``hadm_id``, ``features`` (one value per
    name in ``lab_names``), ``mortality`` (0/1) and the 0/1 group flags
    ``black``, ``female`` and ``public`` (any insurance other than Private).

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import ClinicalPresenceMortalityMIMIC3
        >>> dataset = MIMIC3Dataset(  # doctest: +SKIP
        ...     root="/path/to/mimiciii/1.4", tables=["labevents"]
        ... )
        >>> samples = dataset.set_task(  # doctest: +SKIP
        ...     ClinicalPresenceMortalityMIMIC3()
        ... )
    """


    task_name: str = "ClinicalPresenceMortalityMIMIC3"
    input_schema: ClassVar[dict[str, str]] = {"features": "tensor"}      
    output_schema: ClassVar[dict[str, str]] = {"mortality": "binary"} 

    # Lab variable name -> the MIMIC-III itemids pooled into it.
    

    def __init__(self, lab_names: list[str] | None = None) -> None:
        """Choose which lab variables become features.

        Args:
            lab_names: Names from :data:`PAPER_LAB_GROUPS`, in the order they
                should appear in ``features``. Default: all 67 of the paper.

        Raises:
            ValueError: If a name is not in :data:`PAPER_LAB_GROUPS`.
        """
        super().__init__()
        names = list(PAPER_LAB_GROUPS) if lab_names is None else list(lab_names)
        unknown = [name for name in names if name not in PAPER_LAB_GROUPS]
        if unknown:
            raise ValueError(f"Unknown lab names: {unknown}")
        self.lab_names = names


    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        admissions = patient.get_events(event_type="admissions")
        if not admissions:
            return []
        admission = admissions[-1]  # authors keep each patient's LAST admission

        # Rule: the stay must last at least 1 day (authors: LOS >= 1 day)
        admit = admission.timestamp
        discharge = _parse_time(admission.dischtime) 
        if discharge - admit < timedelta(days=1):
            return []

        # Rule: adults only (authors: AGE > 18, with AGE = days / 365)
        person = patient.get_events(event_type="patients")[0]
        birth = _parse_time(person.dob) 
        if (admit - birth).days / 365 <= 18:
            return []


        # Collect lab readings strictly inside (admit, admit + 24 h)
        window_end = admit + timedelta(days=1)
       
        item_to_lab = {
            str(itemid): name
            for name in self.lab_names
            for itemid in PAPER_LAB_GROUPS[name]
        }

        readings = []  # (time, lab name, value or None)
        for lab in patient.get_events(
            event_type="labevents", start=admit, end=window_end
        ):
            if lab.timestamp <= admit or lab.timestamp >= window_end:
                continue  # authors: Time > 0 and Time < 1 day
            name = item_to_lab.get(str(lab.itemid))
            if name is not None:
                readings.append((lab.timestamp, name, _to_number(lab.valuenum)))

        # Rule: if one lab has 2+ readings at the same time, drop them ALL
        # (authors: duplicated(keep=False) on Patient, Time, Lab)
        times_seen = Counter((time, name) for time, name, _ in readings)

        # Features: most recent non-missing value of each lab
        latest = {}
        for time, name, value in readings:  # time order: later values overwrite
            if times_seen[(time, name)] > 1 or value is None:
                continue
            latest[name] = value
        if not latest:  # authors drop patients with no labs in the window
            return []
        features = [latest.get(name, float("nan")) for name in self.lab_names]

        # Label: died in hospital less than 8 days after admission
        # (authors: DEATHTIME - ADMITTIME < 8 days; PyHealth does not load
        # DEATHTIME, so DISCHTIME of in-hospital deaths stands in for it)
        died_in_hospital = int(admission.hospital_expire_flag) == 1
        died_within_8_days = died_in_hospital and discharge < admit + timedelta(days=8)

        return [
            {
                "patient_id": patient.patient_id,
                "hadm_id": admission.hadm_id,
                "features": features,
                "mortality": int(died_within_8_days),
                "black": int("BLACK" in str(admission.ethnicity).upper()),
                "female": int(person.gender == "F"),
                "public": int(str(admission.insurance).upper() != "PRIVATE"),
            }
        ]

