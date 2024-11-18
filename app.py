from flask import Flask, redirect, request, jsonify
import pickle
import xgboost as xgb
from flasgger import Swagger
import time
import os

PORT = int(os.environ.get('PORT', 5000))


class TokenBucket:
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # Tokens per second
        self.last_refill = time.time()

    def refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        refill_amount = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + refill_amount)
        self.last_refill = now

    def take_token(self):
        self.refill()
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


# Initialize the token bucket
bucket = TokenBucket(capacity=10, refill_rate=3)


def predict_age(
    human={
        'alcohol_consumption=Frequent': 0,
        'alcohol_consumption=Occasional': 0,
        'alcohol_consumption=Unknown': 1,
        'blood_glucose_level_(mg/dl)': 157.65284793866718,
        'bmi': 29.423016908813725,
        'bone_cognitive_combined': 5.854066347148904,
        'bone_density_(g/cm²)': 0.1328682798964727,
        'bone_density_category=Normal': 0,
        'bone_density_category=Osteopenia': 0,
        'bone_density_category=Osteoporosis': 1,
        'bone_density_decline_rate': 0.0014929020213086822,
        'bone_vision_combined': 0.026573655979294543,
        'cholesterol_level_(mg/dl)': 259.46581350104714,
        'chronic_diseases=Diabetes': 0,
        'chronic_diseases=Heart Disease': 0,
        'chronic_diseases=Hypertension': 0,
        'chronic_diseases=Unknown': 1,
        'cognitive_function': 44.05917162252895,
        'diastolic': 109,
        'diet=Balanced': 0,
        'diet=High-fat': 0,
        'diet=Low-carb': 1,
        'diet=Vegetarian': 0,
        'education_level=High School': 0,
        'education_level=Postgraduate': 0,
        'education_level=Undergraduate': 0,
        'education_level=Unknown': 1,
        'family_history=Diabetes': 0,
        'family_history=Heart Disease': 0,
        'family_history=Hypertension': 0,
        'family_history=Unknown': 1,
        'gender=Female': 0,
        'gender=Male': 1,
        'hearing_ability_(db)': 58.78619834245858,
        'hearing_age_interaction': 0.6605190824995346,
        'hearing_category=Mild Loss': 0,
        'hearing_category=Moderate Loss': 0,
        'hearing_category=Normal': 0,
        'hearing_category=Profound Loss': 0,
        'hearing_category=Severe Loss': 1,
        'height_(cm)': 171.14835857585234,
        'income_level=High': 0,
        'income_level=Low': 0,
        'income_level=Medium': 1,
        'medication_use=Occasional': 0,
        'medication_use=Regular': 0,
        'medication_use=Unknown': 1,
        'mental_health_status=Excellent': 0,
        'mental_health_status=Fair': 0,
        'mental_health_status=Good': 1,
        'mental_health_status=Poor': 0,
        'physical_activity_level=High': 0,
        'physical_activity_level=Low': 0,
        'physical_activity_level=Moderate': 1,
        'pollution_exposure': 5.142344384136116,
        'pulse_pressure': 42,
        'sleep_patterns=Excessive': 0,
        'sleep_patterns=Insomnia': 1,
        'sleep_patterns=Normal': 0,
        'smoking_status=Current': 0,
        'smoking_status=Former': 1,
        'smoking_status=Never': 0,
        'stress_levels': 2.797064039425237,
        'sun_exposure': 7.108974826344509,
        'systolic': 151,
        'vision_sharpness': 0.2,
        'weight_(kg)': 86.18519686940489,
    }
):
    """Predicts age based on given human characteristics.

    Args:
        human (dict, optional): A dictionary containing human characteristics. Defaults to a predefined dictionary.

    Returns:
        float: Predicted age.
    """

    input_file = 'model_xgb_eta=0.1_score=1.206.bin'

    with open(input_file, 'rb') as f_in:
        dv, model = pickle.load(f_in)

    # most importante features related to age
    # bone_density_(g/cm²), vision_sharpness, hearing_ability_(db),
    # cognitive_function, cholesterol_level_(mg/dl), blood_glucose_level_(mg/dl),
    # diastolic, systolic, pulse_pressure

    X = dv.transform(human)

    d = xgb.DMatrix(X, feature_names=list(dv.get_feature_names_out()))

    return model.predict(d)[0]


app = Flask(__name__)

# Initialize Swagger
swagger = Swagger(app)

# Rate limiter middleware


@app.before_request
def rate_limiter():
    if not bucket.take_token():
        return jsonify({"detail": "Rate limit exceeded"}), 429


# Define Flask routes
@app.route("/")
def index():
    # Redirect to the Swagger UI
    return redirect("/apidocs/")


@app.route("/predict", methods=['POST'])
def predict():
    """
    Predict age based on clinical history
    ---
    parameters:
      - name: query
        in: body
        required: true
        schema:
          type: object
          properties:
            query:
              type: object
              properties:
                    alcohol_consumption=Frequent:
                      type: integer
                      description: 1 if the person frequently consumes alcohol, 0 otherwise.
                      example: 0
                    alcohol_consumption=Occasional:
                      type: integer
                      description: 1 if the person occasionally consumes alcohol, 0 otherwise.
                      example: 0
                    alcohol_consumption=Unknown:
                      type: integer
                      description: 1 if the alcohol consumption is unknown, 0 otherwise.
                      example: 1
                    blood_glucose_level_(mg/dl):
                      type: number
                      description: Blood glucose level in mg/dl.
                      example: 157.65284793866718
                    bmi:
                      type: number
                      description: Body Mass Index (BMI).
                      example: 29.423016908813725
                    bone_cognitive_combined:
                      type: number
                      description: Combined bone and cognitive health score.
                      example: 5.854066347148904
                    bone_density_(g/cm²):
                      type: number
                      description: Bone mineral density in g/cm².
                      example: 0.1328682798964727
                    bone_density_category=Normal:
                      type: integer
                      description: 1 if bone density is normal, 0 otherwise.
                      example: 0
                    bone_density_category=Osteopenia:
                      type: integer
                      description: 1 if bone density is osteopenic, 0 otherwise.
                      example: 0
                    bone_density_category=Osteoporosis:
                      type: integer
                      description: 1 if bone density is osteoporotic, 0 otherwise.
                      example: 1
                    bone_density_decline_rate:
                      type: number
                      description: Rate of bone density decline.
                      example: 0.0014929020213086822
                    bone_vision_combined:
                      type: number
                      description: Combined bone and vision health score.
                      example: 0.026573655979294543
                    cholesterol_level_(mg/dl):
                      type: number
                      description: Cholesterol level in mg/dl.
                      example: 259.46581350104714
                    chronic_diseases=Diabetes:
                      type: integer
                      description: 1 if the person has diabetes, 0 otherwise.
                      example: 0
                    chronic_diseases=Heart Disease:
                      type: integer
                      description: 1 if the person has heart disease, 0 otherwise.
                      example: 0
                    chronic_diseases=Hypertension:
                      type: integer
                      description: 1 if the person has hypertension, 0 otherwise.
                      example: 0
                    chronic_diseases=Unknown:
                      type: integer
                      description: 1 if the chronic disease status is unknown, 0 otherwise.
                      example: 1
                    cognitive_function:
                      type: number
                      description: Cognitive function score.
                      example: 44.05917162252895
                    diastolic:
                      type: integer
                      description: Diastolic blood pressure.
                      example: 109
                    diet=Balanced:
                      type: integer
                      description: 1 if the diet is balanced, 0 otherwise.
                      example: 0
                    diet=High-fat:
                      type: integer
                      description: 1 if the diet is high-fat, 0 otherwise.
                      example: 0
                    diet=Low-carb:
                      type: integer
                      description: 1 if the diet is low-carb, 0 otherwise.
                      example: 1
                    diet=Vegetarian:
                      type: integer
                      description: 1 if the diet is vegetarian, 0 otherwise.
                      example: 0
                    education_level=High School:
                      type: integer
                      description: 1 if the highest education level is high school, 0 otherwise.
                      example: 0
                    education_level=Postgraduate:
                      type: integer
                      description: 1 if the highest education level is postgraduate, 0 otherwise.
                      example: 0
                    education_level=Undergraduate:
                      type: integer
                      description: 1 if the highest education level is undergraduate, 0 otherwise.
                      example: 0
                    education_level=Unknown:
                      type: integer
                      description: 1 if the education level is unknown, 0 otherwise.
                      example: 1
                    family_history=Diabetes:
                      type: integer
                      description: 1 if diabetes is in family history, 0 otherwise.
                      example: 0
                    family_history=Heart Disease:
                      type: integer
                      description: 1 if heart disease is in family history, 0 otherwise.
                      example: 0
                    family_history=Hypertension:
                      type: integer
                      description: 1 if hypertension is in family history, 0 otherwise.
                      example: 0
                    family_history=Unknown:
                      type: integer
                      description: 1 if family history is unknown, 0 otherwise.
                      example: 1
                    gender=Female:
                      type: integer
                      description: 1 if the gender is female, 0 otherwise.
                      example: 0
                    gender=Male:
                      type: integer
                      description: 1 if the gender is male, 0 otherwise.
                      example: 1
                    hearing_ability_(db):
                      type: number
                      description: Hearing ability in decibels (dB).
                      example: 58.78619834245858
                    hearing_age_interaction:
                      type: number
                      description: Interaction between hearing ability and age.
                      example: 0.6605190824995346
                    hearing_category=Mild Loss:
                      type: integer
                      description: 1 if hearing loss is mild, 0 otherwise.
                      example: 0
                    hearing_category=Moderate Loss:
                      type: integer
                      description: 1 if hearing loss is moderate, 0 otherwise.
                      example: 0
                    hearing_category=Normal:
                      type: integer
                      description: 1 if hearing is normal, 0 otherwise.
                      example: 0
                    hearing_category=Profound Loss:
                      type: integer
                      description: 1 if hearing loss is profound, 0 otherwise.
                      example: 0
                    hearing_category=Severe Loss:
                      type: integer
                      description: 1 if hearing loss is severe, 0 otherwise.
                      example: 1
                    height_(cm):
                      type: number
                      description: Height in centimeters.
                      example: 171.14835857585234
                    income_level=High:
                      type: integer
                      description: 1 if income level is high, 0 otherwise.
                      example: 0
                    income_level=Low:
                      type: integer
                      description: 1 if income level is low, 0 otherwise.
                      example: 0
                    income_level=Medium:
                      type: integer
                      description: 1 if income level is medium, 0 otherwise.
                      example: 1
                    medication_use=Occasional:
                      type: integer
                      description: 1 if medication use is occasional, 0 otherwise.
                      example: 0
                    medication_use=Regular:
                      type: integer
                      description: 1 if medication use is regular, 0 otherwise.
                      example: 0
                    medication_use=Unknown:
                      type: integer
                      description: 1 if medication use is unknown, 0 otherwise.
                      example: 1
                    mental_health_status=Excellent:
                      type: integer
                      description: 1 if mental health status is excellent, 0 otherwise.
                      example: 0
                    mental_health_status=Fair:
                      type: integer
                      description: 1 if mental health status is fair, 0 otherwise.
                      example: 0
                    mental_health_status=Good:
                      type: integer
                      description: 1 if mental health status is good, 0 otherwise.
                      example: 1
                    mental_health_status=Poor:
                      type: integer
                      description: 1 if mental health status is poor, 0 otherwise.
                      example: 0
                    physical_activity_level=High:
                      type: integer
                      description: 1 if physical activity level is high, 0 otherwise.
                      example: 0
                    physical_activity_level=Low:
                      type: integer
                      description: 1 if physical activity level is low, 0 otherwise.
                      example: 0
                    physical_activity_level=Moderate:
                      type: integer
                      description: 1 if physical activity level is moderate, 0 otherwise.
                      example: 1
                    pollution_exposure:
                      type: number
                      description: Pollution exposure level.
                      example: 5.142344384136116
                    pulse_pressure:
                      type: integer
                      description: Pulse pressure.
                      example: 42
                    sleep_patterns=Excessive:
                      type: integer
                      description: 1 if sleep pattern is excessive, 0 otherwise.
                      example: 0
                    sleep_patterns=Insomnia:
                      type: integer
                      description: 1 if sleep pattern is insomnia, 0 otherwise.
                      example: 1
                    sleep_patterns=Normal:
                      type: integer
                      description: 1 if sleep pattern is normal, 0 otherwise.
                      example: 0
                    smoking_status=Current:
                      type: integer
                      description: 1 if smoking status is current, 0 otherwise.
                      example: 0
                    smoking_status=Former:
                      type: integer
                      description: 1 if smoking status is former, 0 otherwise.
                      example: 1
                    smoking_status=Never:
                      type: integer
                      description: 1 if smoking status is never, 0 otherwise.
                      example: 0
                    stress_levels:
                      type: number
                      description: Stress level.
                      example: 2.797064039425237
                    sun_exposure:
                      type: number
                      description: Sun exposure level.
                      example: 7.108974826344509
                    systolic:
                      type: integer
                      description: Systolic blood pressure.
                      example: 151
                    vision_sharpness:
                      type: number
                      description: Vision sharpness.
                      example: 0.2
                    weight_(kg):
                      type: number
                      description: Weight in kilograms.
                      example: 86.18519686940489
    responses:
      200:
        description: Predicted age
        schema:
          type: object
          properties:
            answer:
              type: string
      400:
        description: Bad request due to missing query parameter
        schema:
          type: object
          properties:
            detail:
              type: string
      429:
        description: Rate limit exceeded
        schema:
          type: object
          properties:
            detail:
              type: string
    """
    data = request.json
    query = data.get('query')

    if not query:
        return jsonify({"detail": "Query parameter is required"}), 400

    try:
        answer = predict_age(query)
        return ({"result": str(answer)}), 200
    except Exception as e:
        return jsonify({"detail": str(e)}), 500  # Handle unexpected errors


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=PORT)
