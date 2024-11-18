import pickle
import xgboost as xgb

input_file = 'model_xgb_eta=0.1_score=1.206.bin'

with open(input_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

# most importante features related to age
# bone_density_(g/cm²), vision_sharpness, hearing_ability_(db),
# cognitive_function, cholesterol_level_(mg/dl), blood_glucose_level_(mg/dl),
# diastolic, systolic, pulse_pressure

human = {
    'alcohol_consumption=Frequent': 0,
    'alcohol_consumption=Occasional': 0,
    'alcohol_consumption=Unknown': 1,
    'blood_glucose_level_(mg/dl)': 100.65284793866718,
    'bmi': 50.423016908813725,
    'bone_cognitive_combined': 9.854066347148904,
    'bone_density_(g/cm²)': 10.2328682798964727,
    'bone_density_category=Normal': 1,
    'bone_density_category=Osteopenia': 0,
    'bone_density_category=Osteoporosis': 0,
    'bone_density_decline_rate': 0.5014929020213086822,
    'bone_vision_combined': 0.326573655979294543,
    'cholesterol_level_(mg/dl)': 50.46581350104714,
    'chronic_diseases=Diabetes': 0,
    'chronic_diseases=Heart Disease': 0,
    'chronic_diseases=Hypertension': 0,
    'chronic_diseases=Unknown': 1,
    'cognitive_function': 20.05917162252895,
    'diastolic': 150,
    'diet=Balanced': 0,
    'diet=High-fat': 0,
    'diet=Low-carb': 1,
    'diet=Vegetarian': 0,
    'education_level=High School': 0,
    'education_level=Postgraduate': 0,
    'education_level=Undergraduate': 1,
    'education_level=Unknown': 0,
    'family_history=Diabetes': 0,
    'family_history=Heart Disease': 0,
    'family_history=Hypertension': 1,
    'family_history=Unknown': 0,
    'gender=Female': 1,
    'gender=Male': 0,
    'hearing_ability_(db)': 20.78619834245858,
    'hearing_age_interaction': 0.9605190824995346,
    'hearing_category=Mild Loss': 0,
    'hearing_category=Moderate Loss': 0,
    'hearing_category=Normal': 1,
    'hearing_category=Profound Loss': 0,
    'hearing_category=Severe Loss': 0,
    'height_(cm)': 60.14835857585234,
    'income_level=High': 0,
    'income_level=Low': 1,
    'income_level=Medium': 0,
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
    'pollution_exposure': 10.142344384136116,
    'pulse_pressure': 80,
    'sleep_patterns=Excessive': 0,
    'sleep_patterns=Insomnia': 1,
    'sleep_patterns=Normal': 0,
    'smoking_status=Current': 0,
    'smoking_status=Former': 0,
    'smoking_status=Never': 1,
    'stress_levels': 2.797064039425237,
    'sun_exposure': 7.108974826344509,
    'systolic': 351,
    'vision_sharpness': 0.2,
    'weight_(kg)': 50.18519686940489,
}

X = dv.transform(human)

d = xgb.DMatrix(X, feature_names=list(dv.get_feature_names_out()))

# print(model.feature_names)
# print(dv.get_feature_names_out())

y_pred = model.predict(d)

print(y_pred)
