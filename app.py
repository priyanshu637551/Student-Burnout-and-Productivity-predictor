import streamlit as st
import pickle
import numpy as np

# Load models
reg_model = pickle.load(open("productivity_model.pkl", "rb"))
clf_model = pickle.load(open("burnout_model.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

st.title("Student Burnout & Productivity Predictor")

st.header("Enter today's details")

sleep = st.slider("Sleep Hours", 0.0, 8.0, 4.0)
study = st.slider("Study Hours", 0.0, 8.0, 2.0)
phone = st.slider("Phone Usage Hours", 0.0, 12.0, 6.0)
social = st.slider("Social Media Hours", 0.0, 5.0, 3.0)
mood = st.slider("Mood (1-5)", 1.0, 5.0, 3.0)
caffeine = st.slider("Caffeine Cups", 0, 6, 2)
exercise = st.slider("Exercise Minutes", 0.0, 45.0, 15.0)
classes = st.slider("Classes Attended", 0, 3, 1)
distractions = st.slider("Distractions", 0.0, 40.0, 10.0)
tasks = st.slider("Tasks Completed", 0.0, 10.0, 4.0)

weekend_day = st.checkbox("Is it Weekend?")

burnout_score = (
    phone * 0.5
    + distractions * 0.2
    - sleep * 0.3
    - exercise * 0.1
    + caffeine * 0.2
)



input_data = np.array([[
    sleep, study, phone, social, mood, caffeine,
    exercise, classes, distractions, tasks,
    burnout_score, weekend_day
]])

if st.button("Predict"):

    # Step 1: compute burnout_score
    burnout_score = (
        phone * 0.5
        + distractions * 0.2
        - sleep * 0.3
        - exercise * 0.1
        + caffeine * 0.2
    )

    # Step 2: input for regression (12 features)
    reg_input = np.array([[
        sleep, study, phone, social, mood, caffeine,
        exercise, classes, distractions, tasks,
        burnout_score, weekend_day
    ]])

    productivity_pred = reg_model.predict(reg_input)[0]

    # Step 3: input for classifier (13 features)
    clf_input = np.array([[
        sleep, study, phone, social, mood, caffeine,
        exercise, classes, distractions, tasks,
        burnout_score, productivity_pred,weekend_day
    ]])

    burnout_pred = clf_model.predict(clf_input)[0]
    burnout_label = le.inverse_transform([burnout_pred])[0]

    st.subheader("Results")
    st.write(f"Predicted Productivity Score: {round(productivity_pred,2)}")
    st.write(f"Burnout Level: {burnout_label}")