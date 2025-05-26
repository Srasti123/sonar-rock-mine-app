import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

@st.cache_data
def load_model():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
    df = pd.read_csv(url, header=None)
    X = df.iloc[:, :-1].values
    y = LabelEncoder().fit_transform(df.iloc[:, -1].values)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC(kernel='rbf', C=1, gamma='scale')
    model.fit(X_train, y_train)
    return model

model = load_model()

st.title("Sonar Rock vs Mine Prediction")
st.markdown("Enter 60 sonar signal features below to predict whether it's a Rock or a Mine.")

features = []
for i in range(60):
    val = st.number_input(f"Feature {i+1}", min_value=0.0, max_value=1.0, value=0.5)
    features.append(val)

if st.button("Predict"):
    input_data = np.array(features).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    label = "Mine" if prediction == 1 else "Rock"
    st.success(f"The object is predicted to be a **{label}**")