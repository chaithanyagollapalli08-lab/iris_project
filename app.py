from flask import Flask, render_template, request
import pandas as pd
import pickle
import json

app = Flask(__name__)

# ==============================
# Load Model Safely
# ==============================
def load_model(path):
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
            print(f"{path} loaded successfully")
            return model
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


knn_model = load_model("knn_model.pkl")
nb_model = load_model("nb_model.pkl")


# ==============================
# Load JSON Safely
# ==============================
def load_json(path):
    try:
        with open(path) as f:
            data = json.load(f)
            print(f"{path} loaded successfully")
            return data
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return {}


knn_training_raw = load_json("train_knn.json")
knn_testing_raw = load_json("test_knn.json")
nb_training_raw = load_json("train_NB.json")
nb_testing_raw = load_json("test_NB.json")


# ==============================
# SMART KEY DETECTOR
# ==============================
def normalize_keys(data):
    if not data:
        return {}

    accuracy = None
    confusion_matrix = None
    classification_report = None

    for key, value in data.items():
        k = key.lower()

        if "accuracy" in k:
            accuracy = value

        elif "confusion" in k:
            confusion_matrix = value

        elif "classification" in k:
            classification_report = value

    return {
        "accuracy": accuracy,
        "confusion_matrix": confusion_matrix,
        "classification_report": classification_report
    }


# ==============================
# Species Mapping
# ==============================
species_dict = {
    0: "Iris-setosa",
    1: "Iris-versicolor",
    2: "Iris-virginica"
}


# ==============================
# Home Route
# ==============================
@app.route('/')
def home():
    return render_template(
        "index.html",
        prediction_text=None,
        training_data=None,
        testing_data=None,
        selected_model=None
    )


# ==============================
# Prediction Route
# ==============================
@app.route('/predict', methods=['POST'])
def predict():

    try:
        sl = float(request.form['SepalLengthCm'])
        sw = float(request.form['SepalWidthCm'])
        pl = float(request.form['PetalLengthCm'])
        pw = float(request.form['PetalWidthCm'])

        features = pd.DataFrame(
            [[sl, sw, pl, pw]],
            columns=[
                "SepalLengthCm",
                "SepalWidthCm",
                "PetalLengthCm",
                "PetalWidthCm"
            ]
        )

        model_type = request.form.get("model")

        if model_type == "knn":
            if knn_model is None:
                raise Exception("KNN model not loaded")

            pred = knn_model.predict(features)[0]
            prediction = species_dict.get(pred, pred)

            training = normalize_keys(knn_training_raw)
            testing = normalize_keys(knn_testing_raw)

        elif model_type == "nb":
            if nb_model is None:
                raise Exception("NB model not loaded")

            pred = nb_model.predict(features)[0]
            prediction = species_dict.get(pred, pred)

            training = normalize_keys(nb_training_raw)
            testing = normalize_keys(nb_testing_raw)

        else:
            prediction = "Invalid Model Selected"
            training = None
            testing = None
            model_type = None

    except Exception as e:
        prediction = f"Error: {str(e)}"
        training = None
        testing = None
        model_type = None

    return render_template(
        "index.html",
        prediction_text=prediction,
        training_data=training,
        testing_data=testing,
        selected_model=model_type
    )


# ==============================
# Run App
# ==============================
if __name__ == "__main__":
    app.run(debug=True)
