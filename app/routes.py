import importlib
from flask import Blueprint, render_template, request, abort

from .lab_registry import LABS, get_lab
from .services.model_store import model_store

hub_bp = Blueprint("hub", __name__)

@hub_bp.route("/", methods=["GET"])
def hub():
    # Show all labs on the homepage
    # Optionally also show whether their model file is loadable
    labs_view = []
    for lab in LABS:
        status = "ready"
        error = None
        try:
            model_store.load_bundle(lab.model_path)
        except Exception as e:
            status = "not-ready"
            error = str(e)
        labs_view.append({
            "slug": lab.slug,
            "title": lab.title,
            "description": lab.description,
            "status": status,
            "error": error,
        })

    return render_template("hub.html", labs=labs_view)

@hub_bp.route("/labs/<slug>", methods=["GET", "POST"])
def lab_page(slug: str):
    lab = get_lab(slug)
    if not lab:
        abort(404)

    # Load model bundle
    try:
        bundle = model_store.load_bundle(lab.model_path)
    except Exception as e:
        # Show a nice error page in the lab template
        return render_template(
            lab.template,
            lab=lab,
            error=f"Model not available. Train it first or place the .joblib file.\nDetails: {e}",
            feature_names=[],
            values={},
            prediction=None,
            probabilities=None,
        )

    # Import predict function dynamically
    module_path, func_name = lab.predict_func.split(":")
    mod = importlib.import_module(module_path)
    predict_fn = getattr(mod, func_name)

    # Run prediction logic (handles GET/POST)
    result = predict_fn(request, bundle)

    return render_template(
        lab.template,
        lab=lab,
        **result
    )
