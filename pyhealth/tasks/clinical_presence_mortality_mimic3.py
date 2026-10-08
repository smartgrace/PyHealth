"""Clinical-presence mortality task for MIMIC-III."""

from typing import Any, Dict, List

from .base_task import BaseTask

class ClinicalPresenceMortalityMIMIC3(BaseTask):
    """One sample per hospital admission: label plus three group flags."""

    # The task's name, and what goes IN to the model and what comes OUT
    task_name: str = "ClinicalPresenceMortalityMIMIC3"
    input_schema: Dict[str, str] = {"features": "tensor"}
    output_schema: Dict[str, str] = {"mortality": "binary"}

    # PyHealth calls this once per patient; we return that patient's rows
    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        samples = []
        person = patient.get_events(event_type="patients")[0]
        admissions = patient.get_events(event_type="admissions")
        for admission in admissions:
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "hadm_id": admission.hadm_id,
                    "features": [0.0],  # placeholder; labs come in Lesson 2
                    "mortality": int(admission.hospital_expire_flag),
                    "black": int("BLACK" in str(admission.ethnicity).upper()),
                    "female": int(person.gender == "F"),
                    "public": int(str(admission.insurance).upper() != "PRIVATE"),
                }
            )
        return samples

    