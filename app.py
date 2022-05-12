import streamlit as st
import pandas as pd

from joblib import load

st.title('Heart Failure Prediction')

st.sidebar.markdown("## Predict Housing Price")
st.sidebar.markdown('## User Input')


class HeartFailurePredictor:
    def __init__(self, imported_model, imported_pipeline):
        self.model = imported_model
        self.pipeline = imported_pipeline

    def predict(self, X):
        X_prepared = self.pipeline.transform(X)
        prediction = self.model.predict(X_prepared)
        return prediction[0]


def user_input_features():
    age = st.sidebar.text_input("Age", "40", key='b1').strip()
    sex = st.sidebar.text_input("Sex", "M", key='b2').strip()
    ChestPainType = st.sidebar.text_input("ChestPainType", "ATA", key='b2').strip()
    RestingBP = st.sidebar.text_input("RestingBP", "140", key='b3').strip()
    Cholesterol = st.sidebar.text_input("Cholesterol", "289", key='b4').strip()
    FastingBS = st.sidebar.text_input("FastingBS", "120", key='b59').strip()
    RestingECG = st.sidebar.text_input("RestingECG", "Normal", key='b5').strip()
    MaxHR = st.sidebar.text_input("MaxHR", "140", key='b6').strip()
    ExerciseAngina = st.sidebar.text_input("ExerciseAngina", "N", key='b7').strip()
    Oldpeak = st.sidebar.text_input("Oldpeak", "1.0", key='b8').strip()
    ST_Slope = st.sidebar.text_input("ST_Slope", "Up", key='b9').strip()

    data = {'Age': [int(age)],
            'Sex': [sex],
            'ChestPainType': [ChestPainType],
            'RestingBP': [int(RestingBP)],
            'Cholesterol': [int(Cholesterol)],
            'FastingBS': [int(FastingBS)],
            'RestingECG': [RestingECG],
            'MaxHR': [int(MaxHR)],
            'ExerciseAngina': [ExerciseAngina],
            'Oldpeak': [float(Oldpeak)],
            'ST_Slope': [ST_Slope]
            }

    features = pd.DataFrame(data)
    return features


input_df = user_input_features()

imported_model = load('rf_final_model.joblib')
imported_pipeline = load('pipeline.joblib')


def handle_click():
    st.session_state.age = False
    st.session_state.sex = False
    st.session_state.ChestPainType = False
    st.session_state.RestingBP = False
    st.session_state.Cholesterol = False
    st.session_state.FastingBS = False
    st.session_state.RestingECG = False
    st.session_state.MaxHR = False
    st.session_state.ExerciseAngina = False
    st.session_state.Oldpeak = False
    st.session_state.ST_Slope = False


    predictor = HeartFailurePredictor(imported_model, imported_pipeline)
    test_pred = predictor.predict(input_df)
    st.header("Result")
    st.markdown(f"Prediction result : {'**Chance of Heart Failure**' if test_pred>0.5 else '**No Chance of Heart Failure**'} (Probability: {test_pred : 0.3f}) ")
    st.write("User Input Features")
    st.dataframe(input_df)


st.sidebar.button('Predict', on_click=handle_click)
