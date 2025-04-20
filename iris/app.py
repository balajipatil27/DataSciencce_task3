# iris_app.py
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model and label encoder
model = joblib.load("iris_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form["sepal_length"]),
            float(request.form["sepal_width"]),
            float(request.form["petal_length"]),
            float(request.form["petal_width"])
        ]

        input_df = pd.DataFrame([features], columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
        prediction = model.predict(input_df)[0]
        species = label_encoder.inverse_transform([prediction])[0]

        return render_template("index.html", result=f"Predicted species: {species}")
    except Exception as e:
        return render_template("index.html", result=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
