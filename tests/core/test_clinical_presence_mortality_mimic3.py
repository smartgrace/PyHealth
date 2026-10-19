"""Tests for the clinical-presence mortality task (Jeanselme et al., 2022).

Every test uses small made-up patients, never real MIMIC-III data.
"""

import gzip
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from pyhealth.data import Patient

from pyhealth.tasks.clinical_presence_mortality_mimic3 import (
    PAPER_LAB_GROUPS,
    STRATEGIES,
    ClinicalPresenceMortalityMIMIC3,
    assign_splits,
    impute_cohort,
)

FMT = "%Y-%m-%d %H:%M:%S"
ADMIT = datetime(2150, 1, 1, 8, 0)
GLUCOSE, CREATININE, POTASSIUM = "50931", "50912", "50971"
SMALL_PANEL = ["Glucose", "Creatinine", "Potassium"]


def make_patient(
    admissions=None,
    labs=(),
    gender="F",
    age_years=60,
    patient_id="p1",
):
    """Build a PyHealth Patient directly, without loading any files.

    Args:
        admissions: list of dicts with keys admit, discharge and (optional)
            hadm_id, expire, ethnicity, insurance. Default: one 10-day stay.
        labs: list of (time, itemid, value) tuples.
        gender: "F" or "M".
        age_years: age at the first admission.
        patient_id: id of the patient.
    """
    if admissions is None:
        admissions = [{"admit": ADMIT, "discharge": ADMIT + timedelta(days=10)}]
    birth = admissions[0]["admit"] - timedelta(days=365 * age_years)
    rows = [
        {
            "event_type": "patients",
            "timestamp": None,
            "patients/gender": gender,
            "patients/dob": birth.strftime(FMT),
        }
    ]
    for i, adm in enumerate(admissions):
        rows.append(
            {
                "event_type": "admissions",
                "timestamp": adm["admit"],
                "admissions/hadm_id": adm.get("hadm_id", str(100 + i)),
                "admissions/dischtime": adm["discharge"].strftime(FMT),
                "admissions/hospital_expire_flag": str(adm.get("expire", 0)),
                "admissions/ethnicity": adm.get("ethnicity", "WHITE"),
                "admissions/insurance": adm.get("insurance", "Private"),
            }
        )
    for time, itemid, value in labs:
        rows.append(
            {
                "event_type": "labevents",
                "timestamp": time,
                "labevents/itemid": itemid,
                "labevents/valuenum": None if value is None else str(value),
            }
        )
    frame = pl.from_dicts(rows, infer_schema_length=None)
    return Patient(patient_id=patient_id, data_source=frame)


def hours(n):
    return ADMIT + timedelta(hours=n)


class TestTaskCohortAndFeatures(unittest.TestCase):
    """Patient selection, lab features and labels."""

    def setUp(self):
        self.task = ClinicalPresenceMortalityMIMIC3(lab_names=SMALL_PANEL)

    def test_most_recent_value_is_used(self):
        patient = make_patient(
            labs=[(hours(2), GLUCOSE, 100), (hours(5), GLUCOSE, 150)]
        )
        sample = self.task(patient)[0]
        self.assertEqual(sample["features"][0], 150.0)

    def test_unmeasured_labs_are_nan(self):
        patient = make_patient(labs=[(hours(2), GLUCOSE, 100)])
        features = self.task(patient)[0]["features"]
        self.assertEqual(len(features), 3)
        self.assertTrue(np.isnan(features[1]) and np.isnan(features[2]))

    def test_window_is_strictly_inside_first_24_hours(self):
        patient = make_patient(
            labs=[
                (hours(0), CREATININE, 9.0),  # exactly at admission: ignored
                (hours(23), GLUCOSE, 120),  # inside: kept
                (hours(24), GLUCOSE, 999),  # exactly 24 h: ignored
                (hours(30), POTASSIUM, 9.0),  # after the window: ignored
            ]
        )
        features = self.task(patient)[0]["features"]
        self.assertEqual(features[0], 120.0)
        self.assertTrue(np.isnan(features[1]) and np.isnan(features[2]))

    def test_same_lab_twice_at_same_time_is_dropped(self):
        patient = make_patient(
            labs=[
                (hours(2), GLUCOSE, 100),
                (hours(6), GLUCOSE, 888),
                (hours(6), GLUCOSE, 999),
            ]
        )
        self.assertEqual(self.task(patient)[0]["features"][0], 100.0)

    def test_short_stay_is_excluded(self):
        stay = [{"admit": ADMIT, "discharge": ADMIT + timedelta(hours=20)}]
        patient = make_patient(admissions=stay, labs=[(hours(2), GLUCOSE, 100)])
        self.assertEqual(self.task(patient), [])

    def test_patient_aged_18_or_younger_is_excluded(self):
        patient = make_patient(age_years=18, labs=[(hours(2), GLUCOSE, 100)])
        self.assertEqual(self.task(patient), [])
        patient = make_patient(age_years=19, labs=[(hours(2), GLUCOSE, 100)])
        self.assertEqual(len(self.task(patient)), 1)

    def test_patient_without_labs_is_excluded(self):
        self.assertEqual(self.task(make_patient(labs=[])), [])

    def test_blank_lab_value_counts_as_missing(self):
        patient = make_patient(labs=[(hours(2), GLUCOSE, None)])
        self.assertEqual(self.task(patient), [])

    def test_only_last_admission_is_used(self):
        first = {"admit": ADMIT - timedelta(days=100), "hadm_id": "1",
                 "discharge": ADMIT - timedelta(days=90)}
        last = {"admit": ADMIT, "hadm_id": "2",
                "discharge": ADMIT + timedelta(days=10)}
        patient = make_patient(admissions=[first, last],
                               labs=[(hours(2), GLUCOSE, 100)])
        samples = self.task(patient)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["hadm_id"], "2")

    def test_label_counts_deaths_before_day_8_only(self):
        for days, expected in [(7, 1), (8, 0), (14, 0)]:
            stay = [{"admit": ADMIT, "discharge": ADMIT + timedelta(days=days),
                     "expire": 1}]
            patient = make_patient(admissions=stay, labs=[(hours(2), GLUCOSE, 1)])
            with self.subTest(days=days):
                self.assertEqual(self.task(patient)[0]["mortality"], expected)

    def test_survivor_has_label_zero(self):
        patient = make_patient(labs=[(hours(2), GLUCOSE, 100)])
        self.assertEqual(self.task(patient)[0]["mortality"], 0)

    def test_group_flags(self):
        stay = [{"admit": ADMIT, "discharge": ADMIT + timedelta(days=3),
                 "ethnicity": "BLACK/HAITIAN", "insurance": "Medicaid"}]
        patient = make_patient(admissions=stay, gender="F",
                               labs=[(hours(2), GLUCOSE, 1)])
        sample = self.task(patient)[0]
        self.assertEqual((sample["black"], sample["female"], sample["public"]),
                         (1, 1, 1))
        patient = make_patient(gender="M", labs=[(hours(2), GLUCOSE, 1)])
        sample = self.task(patient)[0]
        self.assertEqual((sample["black"], sample["female"], sample["public"]),
                         (0, 0, 0))


class TestLabPanel(unittest.TestCase):
    """The paper's 67 lab variables (MIMIC-Extract LEVEL1 names)."""

    def test_default_panel_is_the_papers_67_labs(self):
        task = ClinicalPresenceMortalityMIMIC3()
        self.assertEqual(len(task.lab_names), 67)
        self.assertEqual(sum(len(v) for v in PAPER_LAB_GROUPS.values()), 87)
        patient = make_patient(labs=[(hours(2), GLUCOSE, 100)])
        self.assertEqual(len(task(patient)[0]["features"]), 67)

    def test_itemids_of_one_variable_are_pooled(self):
        task = ClinicalPresenceMortalityMIMIC3(lab_names=["Glucose"])
        patient = make_patient(
            labs=[(hours(2), GLUCOSE, 100), (hours(4), "50809", 140)]
        )  # 50809 is blood-gas glucose: same variable, later reading wins
        self.assertEqual(task(patient)[0]["features"], [140.0])

    def test_unknown_lab_name_raises(self):
        with self.assertRaises(ValueError):
            ClinicalPresenceMortalityMIMIC3(lab_names=["Glucose", "Not a lab"])

class TestImputeCohort(unittest.TestCase):
    """Median, MICE and Group MICE imputation."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.x = rng.normal(size=(30, 3))
        self.x[rng.random(self.x.shape) < 0.3] = np.nan
        self.train = np.arange(30) < 24
        self.groups = rng.integers(0, 2, size=(30, 2))

    def test_no_missing_left_and_measured_values_unchanged(self):
        observed = ~np.isnan(self.x)
        for strategy in STRATEGIES: 
            with self.subTest(strategy=strategy):
                out = impute_cohort(self.x, self.train, strategy, groups=self.groups)
                self.assertEqual(out.shape, self.x.shape)
                self.assertFalse(np.isnan(out).any())
                np.testing.assert_array_equal(out[observed], self.x[observed])

    def test_median_learns_from_training_rows_only(self):
        x = np.array([[1.0], [2.0], [3.0], [100.0], [np.nan]])
        train = np.array([True, True, True, False, False])
        self.assertEqual(impute_cohort(x, train, "median")[4, 0], 2.0)

    def test_mice_is_reproducible_with_a_seed(self):
        a = impute_cohort(self.x, self.train, "mice", seed=1)
        b = impute_cohort(self.x, self.train, "mice", seed=1)
        c = impute_cohort(self.x, self.train, "mice", seed=2)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def test_missing_indicators_are_appended(self):
        out = impute_cohort(self.x, self.train, "mice", add_missing_indicators=True)
        self.assertEqual(out.shape, (30, 6))
        np.testing.assert_array_equal(out[:, 3:], np.isnan(self.x).astype(float))

    def test_bad_arguments_raise(self):
        with self.assertRaises(ValueError):
            impute_cohort(self.x, self.train, "mean")
        with self.assertRaises(ValueError):
            impute_cohort(self.x, self.train, "group_mice")


class TestAssignSplits(unittest.TestCase):
    """Patient-level split that copies the authors' structure."""

    def test_sizes_and_every_patient_once(self):
        ids = [str(i) for i in range(100)]
        splits = assign_splits(ids)
        self.assertEqual(set(splits), set(ids))
        counts = {name: list(splits.values()).count(name)
                  for name in ["train", "val", "internal_test", "external_test"]}
        self.assertEqual(counts, {"train": 65, "val": 7, "internal_test": 8,
                                  "external_test": 20})

    def test_same_seed_same_split(self):
        ids = [str(i) for i in range(50)]
        self.assertEqual(assign_splits(ids, seed=3), assign_splits(ids, seed=3))
        self.assertNotEqual(assign_splits(ids, seed=3), assign_splits(ids, seed=4))


def _write_gz(path, header, rows):
    with gzip.open(path, "wt") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")


class TestWithMIMIC3Dataset(unittest.TestCase):
    """End to end through PyHealth's MIMIC-III loader, on 3 fake patients.

    This is the one slower test (a few seconds): it is the only check that the
    task works with the real ``MIMIC3Dataset`` loader and its column names.
    """

    def test_set_task_on_tiny_fake_mimic3(self):
        from pyhealth.datasets import MIMIC3Dataset

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as root, \
                tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cache:
            folder = Path(root)
            _write_gz(folder / "PATIENTS.csv.gz",
                      ["row_id", "subject_id", "gender", "dob", "dod", "dod_hosp",
                       "dod_ssn", "expire_flag"],
                      [[1, 1, "F", "2090-01-01 00:00:00", "", "", "", 0],
                       [2, 2, "M", "2090-01-01 00:00:00", "", "", "", 0],
                       [3, 3, "M", "2140-01-01 00:00:00", "", "", "", 0]])
            _write_gz(folder / "ADMISSIONS.csv.gz",
                      ["row_id", "subject_id", "hadm_id", "admittime", "dischtime",
                       "deathtime", "admission_type", "admission_location",
                       "discharge_location", "insurance", "language", "religion",
                       "marital_status", "ethnicity", "edregtime", "edouttime",
                       "diagnosis", "hospital_expire_flag", "has_chartevents_data"],
                      [[1, 1, 11, "2150-01-01 08:00:00", "2150-01-05 08:00:00", "",
                        "EMERGENCY", "", "", "Medicare", "", "", "", "BLACK", "", "",
                        "", 0, 1],
                       [2, 2, 22, "2150-01-01 08:00:00", "2150-01-03 08:00:00", "",
                        "EMERGENCY", "", "", "Private", "", "", "", "WHITE", "", "",
                        "", 1, 1],
                       [3, 3, 33, "2150-01-01 08:00:00", "2150-01-05 08:00:00", "",
                        "EMERGENCY", "", "", "Private", "", "", "", "WHITE", "", "",
                        "", 0, 1]])
            _write_gz(folder / "ICUSTAYS.csv.gz",
                      ["subject_id", "intime", "icustay_id", "first_careunit",
                       "dbsource", "last_careunit", "outtime"], [])
            _write_gz(folder / "D_LABITEMS.csv.gz",
                      ["row_id", "itemid", "label", "fluid", "category",
                       "loinc_code"],
                      [[1, 50931, "Glucose", "Blood", "Chemistry", ""]])
            _write_gz(folder / "LABEVENTS.csv.gz",
                      ["row_id", "subject_id", "hadm_id", "itemid", "charttime",
                       "value", "valuenum", "valueuom", "flag"],
                      [[1, 1, 11, 50931, "2150-01-01 10:00:00", "110", 110, "", ""],
                       [2, 2, 22, 50931, "2150-01-01 12:00:00", "90", 90, "", ""],
                       [3, 3, 33, 50931, "2150-01-01 12:00:00", "95", 95, "", ""]])

            dataset = MIMIC3Dataset(root=root, tables=["labevents"],
                                    cache_dir=cache)
            samples = dataset.set_task(ClinicalPresenceMortalityMIMIC3(),
                                       num_workers=1)
            by_id = {s["patient_id"]: s for s in samples}

            self.assertEqual(set(by_id), {"1", "2"})  # patient 3 is 10 years old
            glucose = list(PAPER_LAB_GROUPS).index("Glucose")
            self.assertEqual(len(by_id["1"]["features"]), 67)
            self.assertEqual(float(by_id["1"]["features"][glucose]), 110.0)
            self.assertEqual(int(by_id["2"]["mortality"].item()), 1)
            self.assertEqual(by_id["1"]["black"], 1)


if __name__ == "__main__":
    unittest.main()