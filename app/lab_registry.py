from dataclasses import dataclass

@dataclass(frozen=True)
class Lab:
    slug: str
    title: str
    description: str
    template: str
    model_path: str
    predict_func: str  # import path string

LABS = [
    Lab(
        slug="wine",
        title="Wine Cultivar Classifier",
        description="Predict cultivar from chemical features using a trained sklearn model.",
        template="lab/wine.html",
        model_path="labs/wine-categorizer/artifacts/wine-categorizer.joblib",
        predict_func="labs.wine-categorizer.predict:predict_from_form",
    ),
]

def get_lab(slug: str) -> Lab | None:
    for lab in LABS:
        if lab.slug == slug:
            return lab
    return None
