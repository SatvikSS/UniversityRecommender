import streamlit as st
import pandas as pd
import numpy as np
import pickle
import keras

# Load the trained model, scaler, and encoder
with open('classifier_model.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    sc = pickle.load(scaler_file)

with open('encoder.pkl', 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)

# Load the dataset to get feature names for input
df = pd.read_csv('model_data.csv')
x = df.drop('univName', axis=1)
x1 = pd.get_dummies(x)

# Title for the Streamlit app
st.title("University Recommendation System")

# User inputs for recommendations
st.header("Enter your scores for recommendation")

# Create input fields for each feature
research_exp = st.number_input("Research Experience (months)", min_value=0, max_value=100, step=1)
industry_exp = st.number_input("Industry Experience (months)", min_value=0, max_value=100, step=1)
toefl_score = st.number_input("TOEFL Score", min_value=0, max_value=120, step=1)
intern_exp = st.number_input("Internship Experience (months)", min_value=0, max_value=100, step=1)
gre_v = st.number_input("GRE Verbal Score", min_value=130, max_value=170, step=1)
gre_q = st.number_input("GRE Quantitative Score", min_value=130, max_value=170, step=1)
gre_a = st.number_input("GRE Analytical Writing Score", min_value=0.0, max_value=6.0, step=0.1)
cgpa_4 = st.number_input("CGPA (out of 4)", min_value=0.0, max_value=4.0, step=0.1)

# Organize inputs into a dataframe
input_data = {
    "researchExp": research_exp,
    "industryExp": industry_exp,
    "toeflScore": toefl_score,
    "internExp": intern_exp,
    "greV": gre_v,
    "greQ": gre_q,
    "greA": gre_a,
    "cgpa_4": cgpa_4
}

input_df = pd.DataFrame([input_data])

# Preprocess the input data
input_df_encoded = pd.get_dummies(input_df).reindex(columns=x1.columns, fill_value=0)
input_scaled = sc.transform(input_df_encoded)

# Predict the university
if st.button("Recommend University"):
    prediction = classifier.predict(input_scaled)
    predicted_class = np.argmax(prediction, axis=1)
    recommended_university = encoder.inverse_transform(predicted_class)
    st.write("Recommended University:", recommended_university[0])
