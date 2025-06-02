import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf
from catboost import CatBoostClassifier

st.title("Получение предсказаний")

catboost = CatBoostClassifier()
catboost.load_model("models/catboost_classifier.cbm")

scaler = pickle.load(open("models/scaler.pkl", "rb"))
models = {
    'KNN': pickle.load(open('models/knn.pkl', 'rb')),
    'BaggingClassifier': pickle.load(open('models/bagging_classifier.pkl', 'rb')),
    'GradientBoostingClassifier': pickle.load(open('models/grboosting_classifier.pkl', 'rb')),
    'CatBoostClassifier': catboost,
    'StackingClassifier': pickle.load(open('models/stacking_classifier.pkl', 'rb')),
    'Neural Network': tf.keras.models.load_model('models/tf_keras_mlp.keras')
}

uploaded_file = st.file_uploader("Загрузите CSV-файл с данными для предсказания", type=["csv"])

model_choice = st.selectbox("Выберите модель для предсказания", list(models.keys()))

if uploaded_file:
    input_df = pd.read_csv(uploaded_file)
    st.write("Загруженные данные:")
    st.dataframe(input_df)

    model = models[model_choice]

    if st.button("Получить предсказания"):
        if model_choice == "Neural Network":
            features = ['est_diameter_max', 'relative_velocity', 'miss_distance']
            input_scaled = scaler.transform(input_df[features])
            predictions = model.predict(input_scaled)
            predictions = (predictions > 0.5).astype(int).flatten()
        else:
            predictions = model.predict(input_df)
        results_df = pd.DataFrame(predictions, columns=["Predicted"])
        results_df["Predicted"] = results_df["Predicted"].map({
            0: "🟢 Астероид не представляет опасности",
            1: "🔴 Астероид опасен"
        })

        st.write("Предсказанные значения")
        st.write(results_df)

st.markdown("---")
st.subheader("Введите данные вручную для предсказания")

example_features = ['est_diameter_min', 'est_diameter_max', 'relative_velocity', 'miss_distance', 'absolute_magnitude']
feature_descriptions = {
    'est_diameter_min': 'Минимальный расчётный диаметр астероида (км)',
    'est_diameter_max': 'Максимальный расчётный диаметр астероида (км)',
    'relative_velocity': 'Относительная скорость астероида по сравнению с Землёй (км/ч)',
    'miss_distance': 'Расстояние от астероида до Земли (км)',
    'absolute_magnitude': 'Абсолютная звёздная величина астероида'
}

user_input = {}
with st.form("input_form"):
    for feature in example_features:
        label = feature_descriptions.get(feature)
        user_input[feature] = st.number_input(f"{label}:", value=0)
    submitted = st.form_submit_button("Получить предсказание")

if submitted:
    input_single = pd.DataFrame([user_input])
    if model_choice == "Neural Network":
        features = ['est_diameter_max', 'relative_velocity', 'miss_distance']
        input_scaled = scaler.transform(input_single[features])
        prediction = model.predict(input_scaled)
        prediction = (prediction > 0.5).astype(int)[0][0]
    else:
        prediction = model.predict(input_single)[0]
    result = {
        0: "🟢 Астероид не представляет опасности",
        1: "🔴 Астероид опасен"
    }[prediction]

    if prediction == 1:
        st.error(f"Результат: **{result}**")
    else:
        st.success(f"Результат: **{result}**")
