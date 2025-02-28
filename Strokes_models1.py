import numpy as np
import pickle # to load the model
import streamlit as st
import pandas as pd
import requests


# Loading the saved model copy the loaded_model line of code from jupyter notebook
# copy the path to where the loaded model is savel
# change the \ to /
#loaded_model = pickle.load(open('E:\Ezekiel\Model_Deployment/trained_model1.sav', 'rb'))  
#loaded_model = pickle.load(open('https://github.com/ezekielmose/Machine-Learning/blob/main/trained_model1.sav', 'rb')) 

#import pickle


# URL of the .sav file
#url = 'https://github.com/ezekielmose/Machine-Learning/raw/main/trained_model1.sav'

url = "https://raw.githubusercontent.com/ezekielmose/Stroke_model/refs/heads/main/strock_model_new.pkl"

# Download the file
#loaded_model1 = requests.get(url)

# Function to download and load the model
def load_model():
    response = requests.get(url)
    with open("strock_model_new.pkl", "wb") as f:
        f.write(response.content)
    with open("strock_model_new.pkl", "rb") as f:
        model = pickle.load(f)
    return model


model = load_model()
# Now, you can use the loaded model for predictions


# creating a function for prediction
def st_prediction(input_data):

    ## Copy from Jupyter, the code for the unstandadized data 
    ## changing input data to numpy array because processing is easier than list 
    input_data_as_numpy_array= np.array(input_data)
    # reshaping the array for predicting 
    
    # Prepare the input data as an array or DataFrame (depending on your model)
    # input_data = [age, sex, Chest_Pain, Blood_Pressure]
    input_data_reshaped = np.array(input_data).reshape(1, -1)
    
   #  input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    
    print(prediction)
    if prediction [0] == 0:
        return "The Person Does not have a st" # insted of print change to return
    else:
        return "The Person has st" # insted of print change to return  
    
# Streamlit library to craete a user interface   
def main():
    st.title ("Stroke Prediction Model")
    
    gender = st.text_input("What is the Gender (0 - Female and 1 - male)")
    age = st.text_input("Enter the age")
    hypertension = st.text_input(" Hypertension 0 for -ve and 1 for +ve")
    heart_disease = st.text_input(" Heart_disease 0 for has and 1 for not ")
    ever_married = st.text_input(" Ever_married 0 for No and 1 for Yes")
    work_type = st.text_input(" Work_type 0 for private and 1 for self employded, 2 for children, 3 for gov job, and 4 for Never_worked")
    Residence_type = st.text_input ("Residence_type 0 for ubarn  and 1 for rural")
    avg_glucose_level = ("Enter any value of (avg_glucose_level) as per the measurements")
    bmi =st.text_input ("Enter any value of (BMI) as per the measurements")
    smoking_status = st.text_input("Smoking_status 0 for never smoked, 1 for Unknown, 2 for formerly smoked, 3 for smokes")
    
    
    ## Numeric conversion
    gender = pd.to_numeric(gender, errors='coerce') 
    age = pd.to_numeric(age, errors='coerce')
    hypertension = pd.to_numeric(hypertension, errors='coerce')
    heart_disease = pd.to_numeric(heart_disease, errors='coerce')
    ever_married = pd.to_numeric(ever_married, errors='coerce')
    Residence_type = pd.to_numeric(Residence_type, errors='coerce')
    avg_glucose_level = pd.to_numeric(avg_glucose_level, errors='coerce')
    bmi = pd.to_numeric(bmi, errors='coerce')
    smoking_status = pd.to_numeric(smoking_status, errors='coerce')

    # code for prediction
    diagnosis = '' # string tha ontaons null values whose values are stored in the prediction
    
    # creating  a prediction button
    if st.button("PREDICT"):
        diagnosis = st_prediction ([gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi,smoking_status])    
    st.success(diagnosis)
    
 
# this is to allow our web app to run from anaconda command prompt where the cmd takes the main() only and runs the code
if __name__ == '__main__':
    main()
