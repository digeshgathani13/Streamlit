import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


st.write("""
    #Diabetes Detection
    Detect someone has diabetes using Machine Learning and Python !.
""")
path="C://Digesh Personal//SimpliLearn//Capstone_Project//Streamlit//"
image = Image.open(path+"ML.jpg")
st.image(image, caption="ML", use_column_width=True)

df = pd.read_csv(path+"health care diabetes.csv")
st.subheader("Data Information")
st.dataframe(df)
st.write(df.describe())
chart = st.bar_chart(df)
X = df.drop(["Outcome"], axis = 1)
Y = df["Outcome"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

def get_user_input():   
    Pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 3)
    Glucose = st.sidebar.slider("Glucose", 0, 199, 117)
    BloodPressure = st.sidebar.slider("BloodPressure", 0, 122, 72)
    SkinThickness = st.sidebar.slider("SkinThickness", 0, 99, 23)
    Insulin = st.sidebar.slider("Insulin", 0.0, 846.0, 30.0)
    BMI = st.sidebar.slider("BMI", 0.0, 67.1, 32.0)
    DPF = st.sidebar.slider("DPF", 0.078, 2.42, 0.3275)
    Age = st.sidebar.slider("Age", 21, 80, 29)
    user_data = {}
    user_data = {
        "Pregnancies": Pregnancies,
        "Glucose": Glucose,
        "BloodPressure": BloodPressure,
        "SkinThickness": SkinThickness,
        "Insulin": Insulin,
        "BMI": BMI,
        "DPF": DPF,
        "Age": Age
    }
    
    features = pd.DataFrame(user_data, index = [0])
    return features

user_input = get_user_input()

st.subheader("User Input:")
st.write(user_input)

rfc = RandomForestClassifier()
rfc.fit(X_train,Y_train)

st.subheader("model test accuracy score:")
st.write(str(accuracy_score(Y_test, rfc.predict(X_test)) * 100), "%")

predict = rfc.predict(user_input)
st.subheader("Classification:")
st.write(predict)