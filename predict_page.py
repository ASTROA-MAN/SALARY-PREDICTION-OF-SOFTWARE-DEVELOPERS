import streamlit as st
import pickle
import numpy as np


def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

def show_predict_page():
    st.title("SOFTWARE DEVELOPERS SALARY PREDICTION")

    st.write("""### Enter The Information To Predict The Salary""")


    countries = (
        'United States of America',
        'India',
        'Germany',
        'United Kingdom of Great Britain and Northern Ireland',
        'Canada',
        'France',
        'Brazil',
        'Spain',
        'Netherlands',
        'Australia',
        'Poland',
        'Italy',
        'Russian Federation',
        'Sweden',
        'Turkey',
        'Switzerland',
        'Israel',
        'Norway',
    )

    education = (
        'Less than a Bachelors',
        'Bachelor’s degree',
        'Master’s degree',
        'Post grad',
    )

    country = st.selectbox('COUNTRY', countries)
    education = st.selectbox('EDUCATION LEVEL', education)
    experience = st.slider('YEARS OF EXPERIENCE', 0, 50, 3)

    ok = st.button('CALCULATE SALARY')
    if ok:
        X = np.array([[country, education, experience]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X = X.astype(float)

        salary = regressor.predict(X)

        st.subheader(f'The Estimated Salary Is ${salary[0]:.2f}')

