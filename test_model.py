import joblib
import pandas as pd

# Load model and feature columns
model = joblib.load("heart_model.pkl")
model_columns = joblib.load("model_columns.pkl")

print("\n--- HEART FAILURE RISK PREDICTION (TERMINAL MODE) ---\n")

def get_value(name, default):
    value = input(f"Enter {name} (Press Enter to use default = {default}): ")
    return float(value) if value.strip() != "" else default

# Collect inputs same as training order
age = get_value("Age (years)", 60)
cpk = get_value("CPK - Creatinine Phosphokinase (mcg/L)", 200)
ejection_fraction = get_value("Ejection Fraction (%)", 40)
platelets = get_value("Platelets (kiloplatelets/mL)", 250000)
serum_creatinine = get_value("Serum Creatinine (mg/dL)", 1.2)
serum_sodium = get_value("Serum Sodium (mEq/L)", 135)
sex = get_value("Sex (1 = Male, 0 = Female)", 1)
time = get_value("Follow-up Time (days)", 60)

# Prepare input in DataFrame format
user_data = pd.DataFrame([[age, cpk, ejection_fraction, platelets,
                           serum_creatinine, serum_sodium, sex, time]],
                         columns=model_columns)

# Predict
prediction = model.predict(user_data)[0]

# Show Result
if prediction == 1:
    print("\nRESULT: HIGH RISK ⚠️ (Patient may face Heart Failure)\n")
else:
    print("\nRESULT: LOW RISK ✅ (Patient condition appears stable)\n")
