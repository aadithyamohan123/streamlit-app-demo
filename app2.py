import streamlit as st
import pandas as pd
import pickle

def main():
    st.sidebar.title("Inputs")
    
    # Collecting input data from the user via the sidebar
    soil_type = st.sidebar.radio("Soil Type", ['loam', 'sandy', 'clay'])
    water_freq = st.sidebar.radio("Water Frequency", ['bi-weekly', 'weekly', 'daily'])
    fertilizer_type = st.sidebar.radio("Fertilizer Type", ['chemical', 'organic', 'none'])
    sunlight_hrs = st.sidebar.slider("Select the Sunlight Hours", 3, 10, 5)
    temp = st.sidebar.slider("Select the Temperature", 10, 40, 15)
    humidity = st.sidebar.slider("Select the Humidity", 30, 80, 40)
    
    st.title("Plant Growth Prediction")
    
    # Loading the first model (Logistic Regression)
    with open('model2.pkl', 'rb') as f:
        model1 = pickle.load(f)
    
    # Loading the second model (Naive Bayes)
    with open('model3.pkl', 'rb') as f:
        model2 = pickle.load(f)
    
    if st.button("Predict"):
        # Creating a DataFrame for input
        inp = pd.DataFrame({
            'Soil_Type': [soil_type],
            'Water_Frequency': [water_freq],
            'Fertilizer_Type': [fertilizer_type],
            'Sunlight_Hours': [sunlight_hrs],
            'Temperature': [temp],
            'Humidity': [humidity]
        })

        # Predicting with the first model (Logistic Regression)
        prediction1 = model1.predict_proba(inp)
        st.write(f"Prediction probabilities (Logistic Regression): {prediction1}")
        
        # Predicting with the second model (Naive Bayes)
        prediction2 = model2.predict_proba(inp)
        st.write(f"Prediction probabilities (Naive Bayes): {prediction2}")

if __name__ == "__main__":
    main()


