
import streamlit as st
import pandas as pd
import joblib

# Load the trained model and feature columns
model = joblib.load("student_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("Student Performance Prediction")
st.write("Predict a student's exam score using student-related factors.")

st.sidebar.header("Enter Student Details")

hours = st.sidebar.slider("Hours Studied", 0, 12, 5)
attendance = st.sidebar.slider("Attendance (%)", 0, 100, 80)
previous = st.sidebar.slider("Previous Scores", 0, 100, 70)
sleep = st.sidebar.slider("Sleep Hours", 0, 12, 7)
tutoring = st.sidebar.slider("Tutoring Sessions", 0, 10, 2)
physical = st.sidebar.slider("Physical Activity (hours)", 0, 10, 3)

parental = st.sidebar.selectbox(
    "Parental Involvement", ["Low", "Medium", "High"]
)
resources = st.sidebar.selectbox(
    "Access to Resources", ["Low", "Medium", "High"]
)
extra = st.sidebar.selectbox(
    "Extracurricular Activities", ["Yes", "No"]
)
motivation = st.sidebar.selectbox(
    "Motivation Level", ["Low", "Medium", "High"]
)
internet = st.sidebar.selectbox("Internet Access", ["Yes", "No"])
income = st.sidebar.selectbox(
    "Family Income", ["Low", "Medium", "High"]
)
teacher = st.sidebar.selectbox(
    "Teacher Quality", ["Low", "Medium", "High"]
)
school = st.sidebar.selectbox("School Type", ["Public", "Private"])
peer = st.sidebar.selectbox(
    "Peer Influence", ["Positive", "Negative", "Neutral"]
)
disability = st.sidebar.selectbox(
    "Learning Disabilities", ["Yes", "No"]
)
education = st.sidebar.selectbox(
    "Parental Education Level",
    ["High School", "College", "Postgraduate"]
)
distance = st.sidebar.selectbox(
    "Distance from Home", ["Near", "Moderate", "Far"]
)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

if st.button("Predict Exam Score"):
    # Start with all expected encoded features set to zero
    input_data = pd.DataFrame(
        0, index=[0], columns=feature_columns
    )

    # Set numeric features
    numeric_values = {
        "Hours_Studied": hours,
        "Attendance": attendance,
        "Previous_Scores": previous,
        "Sleep_Hours": sleep,
        "Tutoring_Sessions": tutoring,
        "Physical_Activity": physical,
    }

    for col, value in numeric_values.items():
        if col in input_data.columns:
            input_data.loc[0, col] = value

    # Set categorical features using the training column names
    categorical_values = {
        "Parental_Involvement": parental,
        "Access_to_Resources": resources,
        "Extracurricular_Activities": extra,
        "Motivation_Level": motivation,
        "Internet_Access": internet,
        "Family_Income": income,
        "Teacher_Quality": teacher,
        "School_Type": school,
        "Peer_Influence": peer,
        "Learning_Disabilities": disability,
        "Parental_Education_Level": education,
        "Distance_from_Home": distance,
        "Gender": gender,
    }

    for feature, value in categorical_values.items():
        col = f"{feature}_{value}"
        if col in input_data.columns:
            input_data.loc[0, col] = 1

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Exam Score: {prediction:.2f}")