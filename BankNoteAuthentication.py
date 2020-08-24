# BankNoteAuthentication

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_option('deprecation.showfileUploaderEncoding', False)
w = st.file_uploader("Upload a CSV file", type="csv")
if w:
    import pandas as pd

    df = pd.read_csv(w)
    st.write(df)
    
    st.write(df.describe())
    df['class'].value_counts().plot(kind='bar')
    st.pyplot()
    #st.bar_chart(df['class'])
    chart = st.bar_chart(df)
    X = df.drop(["class"], axis = 1)
    y = df["class"]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    def get_user_input():   
        variance = st.sidebar.slider("variance", -10.0, 10.0, 3.0)
        skewness = st.sidebar.slider("skewness", -15.0, 15.0, 3.0)
        curtosis = st.sidebar.slider("curtosis", -20.0, 20.0, 2.0)
        entropy = st.sidebar.slider("entropy", -10.0, 10.0, 2.0)
        user_data = {}
        user_data = {
            "variance": variance,
            "skewness": skewness,
            "curtosis": curtosis,
            "entropy": entropy
        }
        
        features = pd.DataFrame(user_data, index = [0])
        return features

    user_input = get_user_input()
    st.subheader("User Input:")
    st.write(user_input)

    from sklearn.preprocessing import StandardScaler
    sc= StandardScaler()
    X_train=sc.fit_transform(X_train)
    X_test=sc.transform(X_test)

    from sklearn.ensemble import RandomForestRegressor
    rfc=RandomForestRegressor(n_estimators=20,random_state= 0)
    rfc.fit(X_train,Y_train)
    y_pred = rfc.predict(X_test)
    st.subheader("model test accuracy score:")
    st.write(str(accuracy_score(Y_test, y_pred.round()) * 100), "%")

    user_input = sc.transform(user_input)
    predict = rfc.predict(user_input)
    st.subheader("Classification:")
    st.write(predict.round())
