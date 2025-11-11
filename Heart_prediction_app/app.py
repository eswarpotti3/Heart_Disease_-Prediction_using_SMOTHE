from flask import Flask, render_template, request, make_response
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-store"
    return response

model = joblib.load("heart_model.pkl")
model_columns = joblib.load("model_columns.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":

        def val(field, default):
            x = request.form.get(field)
            return float(x) if x and x.strip() != "" else default

        age = val("age", 60)
        cpk = val("creatinine_phosphokinase", 200)
        ejection_fraction = val("ejection_fraction", 40)
        platelets = val("platelets", 250000)
        serum_creatinine = val("serum_creatinine", 1.2)
        serum_sodium = val("serum_sodium", 135)
        sex = val("sex", 1)
        time = val("time", 60)

        user_data = pd.DataFrame([[age, cpk, ejection_fraction, platelets, serum_creatinine, serum_sodium, sex, time]],
                                 columns=model_columns)

        prediction = model.predict(user_data)[0]

        result = "HIGH RISK ⚠️ (Patient may face Heart Failure)" if prediction == 1 else "LOW RISK ✅ (Stable Condition)"
        return render_template("index.html", result=result)

    # ✅ On refresh or first load → show empty form
    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
