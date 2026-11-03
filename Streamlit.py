import pickle
import streamlit as st
from streamlit_option_menu import option_menu

#loading the saved models

diabetes_model = pickle.load(open('I:/Diabetes prediction/diabetes_model.sav', 'rb'))

# Sidebar for navigation

with st.sidebar:
    selected = option_menu('Diabetes Prediction', ['Diabetes Prediction'],
    icons= ['activity','Heart pulse'], 
    default_index=0 )


#Diabetes page 

if(selected== 'Diabetes Prediction'):
    #page title
    st.title('Diabetes Prediction using ML')
    
#for columns 
    col1,col2,col3 = st.columns(3)
    
    with col1:
     Pregnancies = st.text_input('Number of pregnancies') 
    
    with col2:
     Glucose= st.text_input('Glucose Level')
     
    with col3:
     BloodPressure = st.text_input('BloodPressure Value')
     
    with col1:
     SkinThickness = st.text_input('SkinThickness Value')
     
    with col2: 
     Insulin = st.text_input('Insulin Level')
     
    with col3: 
     BMI = st.text_input('BMI Value')
     
    with col1: 
     DiabetesPedigreeFunction = st.text_input('DiabetesPedigreeFunction Value')
     
    with col2: 
     Age = st.text_input('Age of the person')
    
#Code for Prediction

diab_diagnosis = ''

#Creating button for Result

if st.button('Diabetes Test Result'):
    diab_prediction = diabetes_model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness, Insulin,  BMI, DiabetesPedigreeFunction, Age  ]])
    
    if diab_prediction[0] ==1:
        diab_diagnosis = 'The Person is Diabetic'
    else:
        diab_diagnosis = 'The person is Non Diabetic'
       
st.success(diab_diagnosis)
   
    
    
    
#Heart Disease Page 
    
# if(selected== 'Heart Disease prediction'):
#     #page title
#     st.title('Heart Disease Prediction using ML')
    
    
    
    
    
    
        
        

 

