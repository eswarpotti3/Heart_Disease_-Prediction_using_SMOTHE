
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=BASE_DIR)
model = joblib.load(os.path.join(BASE_DIR, "heart_model.pkl"))

@app.get("/", response_class=HTMLResponse)
def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/", response_class=HTMLResponse)
def predict(
    request: Request,
    age: float = Form(...),
    creatinine_phosphokinase: float = Form(...),
    ejection_fraction: float = Form(...),
    platelets: float = Form(...),
    serum_creatinine: float = Form(...),
    serum_sodium: float = Form(...),
    sex: int = Form(...),
    time: float = Form(...)
):
    data = np.array([[age, creatinine_phosphokinase, ejection_fraction, platelets,
                      serum_creatinine, serum_sodium, sex, time]])
    prediction = model.predict(data)[0]
    result = "High Risk of Heart Failure" if prediction == 1 else "Low Risk of Heart Failure"
    return templates.TemplateResponse("index.html", {
    "request": request,
    "result": result,
    "age": age,
    "creatinine_phosphokinase": creatinine_phosphokinase,
    "ejection_fraction": ejection_fraction,
    "platelets": platelets,
    "serum_creatinine": serum_creatinine,
    "serum_sodium": serum_sodium,
    "sex": str(sex),
    "time": time
})
