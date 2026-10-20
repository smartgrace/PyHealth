pyhealth.tasks.clinical_presence_mortality_mimic3
=================================================

Mortality prediction on MIMIC-III for studying how imputation of missing
laboratory tests affects group fairness, reproducing Jeanselme et al. (2022),
"Imputation Strategies Under Clinical Presence: Impact on Algorithmic
Fairness" (ML4H, PMLR 193).

The task builds one sample per adult patient (last hospital admission lasting
at least 24 hours): the most recent value of each of the paper's 67 lab
variables in the first 24 hours (``NaN`` when not measured), death within
8 days of admission, and Black / female / public-insurance flags.
:func:`impute_cohort` fills the missing labs with the paper's strategies
(Median, MICE, Group MICE, with optional missingness indicators) or the
MissForest extension, and :func:`assign_splits` reproduces the authors'
patient split. See
``examples/mimic3_clinical_presence_mortality_logistic_regression.py``.

.. autoclass:: pyhealth.tasks.clinical_presence_mortality_mimic3.ClinicalPresenceMortalityMIMIC3
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.tasks.clinical_presence_mortality_mimic3.impute_cohort

.. autofunction:: pyhealth.tasks.clinical_presence_mortality_mimic3.assign_splits

.. autodata:: pyhealth.tasks.clinical_presence_mortality_mimic3.PAPER_LAB_GROUPS
    :no-value:

.. autodata:: pyhealth.tasks.clinical_presence_mortality_mimic3.STRATEGIES