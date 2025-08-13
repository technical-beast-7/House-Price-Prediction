# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load trained model
with open("house_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Get form values
            data = [
                float(request.form['bedrooms']),
                float(request.form['bathrooms']),
                float(request.form['sqft_living']),
                float(request.form['floors']),
                float(request.form['waterfront']),
                float(request.form['view']),
                float(request.form['condition']),
                float(request.form['grade'])
            ]
            data = np.array(data).reshape(1, -1)
            prediction = model.predict(data)[0]
            prediction = round(prediction, 2)
        except:
            prediction = "Invalid input!"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
