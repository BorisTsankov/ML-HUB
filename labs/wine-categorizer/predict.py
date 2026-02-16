import pandas as pd

def predict_from_form(request, bundle: dict):
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    classes = bundle["classes"]

    prediction = None
    probabilities = None
    error = None

    # Keeps form values after submit
    values = {f: "" for f in feature_names}

    if request.method == "POST":
        try:
            x = {}
            missing = []
            invalid = []

            for f in feature_names:
                raw = request.form.get(f, "").strip()

                # Allow EU decimals (1,92 -> 1.92)
                raw = raw.replace(",", ".")
                values[f] = raw

                if raw == "":
                    missing.append(f)
                    continue

                try:
                    x[f] = float(raw)
                except ValueError:
                    invalid.append(f)

            if missing:
                error = "Missing values for: " + ", ".join(missing)
            elif invalid:
                error = "Invalid numeric values for: " + ", ".join(invalid)
            else:
                X = pd.DataFrame([x], columns=feature_names)

                proba = model.predict_proba(X)[0]
                best_idx = int(proba.argmax())

                prediction = classes[best_idx]
                probabilities = list(zip(classes, proba))
                probabilities.sort(key=lambda t: t[1], reverse=True)

        except Exception as e:
            error = f"Unexpected error: {str(e)}"

    return {
        "error": error,
        "feature_names": feature_names,
        "values": values,
        "prediction": prediction,
        "probabilities": probabilities,
    }
