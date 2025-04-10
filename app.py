# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 11:12:00 2025

@author: davex
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import streamlit as st 

#import new modules
import src.credit_eligibility_model as credit_eligibility_model
import src.real_estate_solution_model as real_estate_solution_model
import src.ucla_neural_networks_model as ucla_neural_networks_model
import src.unsupervised_clustering_model as unsupervised_clustering_model



# Set the page title and description
st.set_page_config(
        page_title="CST 2216 Term Project",
    )

st.title("Credit Loan Eligibility Predictor")
st.image("./images/pawnbrokers-symbol-logo-png_seeklogo-106731.png", width=150)
st.write("""
This app predicts whether a loan applicant is eligible for a loan 
based on various personal and financial characteristics.
""")

# Optional password protection (remove if not needed)
#password_guess = st.text_input("Please enter your password?")
# this password is stores in streamlit secrets
#if password_guess != st.secrets["password"]:
#    st.stop()


credit_eligibility_model.main()

# Load the pre-trained model
rf_pickle = open("./data/random_forest_credit.pickle", "rb")
rf_model = pickle.load(rf_pickle)
rf_pickle.close()


#TO DO: Add code to allow user to input property parameters for price prediction

# Prepare the form for individual predictions
with st.form("user_inputs"):
    st.subheader("Loan Applicant Details")
    
    # Gender input
    Gender = st.selectbox("Gender", options=["Male", "Female"])
    
    # Marital Status
    Married = st.selectbox("Marital Status", options=["Yes", "No"])
    
    # Dependents
    Dependents = st.selectbox("Number of Dependents", 
                               options=["0", "1", "2", "3+"])
    
    # Education
    Education = st.selectbox("Education Level", 
                              options=["Graduate", "Not Graduate"])
    
    # Self Employment
    Self_Employed = st.selectbox("Self Employed", options=["Yes", "No"])
    
    # Applicant Income
    ApplicantIncome = st.number_input("Applicant Monthly Income", 
                                       min_value=0, 
                                       step=1000)
    
    # Coapplicant Income
    CoapplicantIncome = st.number_input("Coapplicant Monthly Income", 
                                         min_value=0, 
                                         step=1000)
    
    # Loan Amount
    LoanAmount = st.number_input("Loan Amount", 
                                  min_value=0, 
                                  step=1000)
    
    # Loan Amount Term
    Loan_Amount_Term = st.selectbox("Loan Amount Term (Months)", 
                                    options=["360","240", "180", "120", "60"])
    
    # Credit History
    Credit_History = st.selectbox("Credit History", 
                                  options=["1", "0"])

    
    # Property Area
    Property_Area = st.selectbox("Property Area", 
                                 options=["Urban", "Semiurban", "Rural"])
    
    # Submit button
    submitted = st.form_submit_button("Predict Loan Eligibility")

Gender_Male = 0
if Gender == "Male":
    Gender_Male = 1
elif Gender == "Female":
    Gender_Male = 0

Married_Yes = 0
if Married == "Yes":
    Married_Yes = 1
elif Married == "No":
    Married_Yes = 0

Dependents_1,Dependents_2,Dependents_3 = 0,0,0
if Dependents == 1:
    Dependents_1 = 1
elif Dependents == 2:
    Dependents_2 = 1
elif Dependents == 3:
    Dependents_3 = 1

Education_Not_Graduate = 0
if Education == "Graduate":
    Education_Not_Graduate = 0
elif Education == "Not_Graduate":
    Education_Not_Graduate = 1

Self_Employed_Yes = 0
if Self_Employed == "Yes":
    Self_Employed_Yes = 1
elif Self_Employed == "No":
    Self_Employed_Yes = 0


Property_Area_Semiurban, Property_Area_Urban = 0,0
if Property_Area == "Semiurban":
    Property_Area_Semiurban = 1
elif Property_Area == "Urban":
    Property_Area_Urban == 1
    

new_prediction = rf_model.predict(
    [[ Dependents, ApplicantIncome, CoapplicantIncome, LoanAmount,
       Loan_Amount_Term, Credit_History, Gender_Male, Married_Yes,
       Education_Not_Graduate, Self_Employed_Yes,
       Property_Area_Semiurban, Property_Area_Urban
     ]]
)

st.subheader("Predicting the outcome:")
# predicted_outcome = new_prediction[0]
if new_prediction[0] == 1:
    #testing only
    print("You are eligible")
    st.write("You are eligible")
elif new_prediction[0] == 0:
    #testing only
    print("Sorry, you are not eligible for loan")
    st.write("Sorry, you are not eligible for loan")

st.write(
    """We used a machine learning (Random Forest) model to predict your eligibility, the features used in this prediction are ranked by relative
    importance below."""
)
st.image("./images/feature_importance.png")


#Begin Real Estate Price Prediction Demo
st.title("Real Estate Price Prediction Demo")
st.image("./images/real-estate.jpg", width=150)
st.header("""
        This app uses decision tree or random forest models to predict real estate prices based on user input.
        """)


year_sold = st.text_input("Year")
    
property_tax = st.number_input("Property Tax")
    
insurance = st.number_input("Insurance")
    
beds = st.number_input("No. of Beds")
    
baths = st.number_input("No. of Bathrooms")
    
sqft = st.number_input("Area in Square Feet")
    
lot_size = st.number_input("Lot Size in Square Feet")

year_built = st.text_input("Year Built")
        
basement_value = st.selectbox("Basement", ('Yes', 'No'))
if(basement_value == 'Yes'):
    basement = 1
else:
    basement = 0
        
popular_value = st.selectbox("Popular: ", ('Yes', 'No'))
    
if(popular_value == 'Yes'):
    popular = 1
else:
    popular = 0


recession_value = st.selectbox("Recession ", ('Yes', 'No'))

if(recession_value == 'Yes'):
    recession = 1
else:
    recession = 0
    
property_age = st.number_input("Property Age")
    
property_type = st.radio("Property Type: ", ('Bunglow', 'Condo'))
    
if(property_type == 'Bunglow'):
        property_type_Bunglow = 1
        property_type_Condo = 0
else:
        property_type_Condo = 1
        property_type_Condo = 0
        
methodology = st.radio("Select methodology: ", ("Decision Tree", "Random Forest"))

if methodology == "Decision Tree":
    st.success("Predicting property price with decision tree!")
else:
    st.success("Predicting property price with random forest!")
        

if(st.button("Predict Price")):
    if methodology == "Decision Tree":
        dt_price =  real_estate_solution_model.DecisionTreeApp(year_sold, property_tax, insurance, beds, baths, sqft, year_built, lot_size, basement, popular, recession, property_age, property_type_Bunglow, property_type_Condo)
        formatted =  "${:,.2f}".format(dt_price[0])
        #st.write(formatter(dt_price))
        st.write(formatted)
        #st.write(dt_price['value'].map("${:,.2f}").format)         
    else:
        rf_price =  real_estate_solution_model.RandomForestApp(year_sold, property_tax, insurance, beds, baths, sqft, year_built, lot_size, basement, popular, recession, property_age, property_type_Bunglow, property_type_Condo)
        formatted = "${:,.2f}".format(rf_price[0])
        #st.write(formatter(rf_price))
        st.write(formatted)
        #st.write(rf_price['value'].map("${:,.2f}").format)      
        
          
          
st.title("UCLA Neural Networks Model Demo")
st.write("""
    This app demonstrates a neural network model being used to predict the probability of admission to UCLA based on GRE and TOEFL scores. The user can define the number of hidden units for each layer, batch size, and maximum number of iterations.
    """)

layer1_units = int(st.number_input("Number of hidden units in Layer #1", 0, 100, "min", 1))
layer2_units = int(st.number_input("Number of hidden units in Layer #2",  0, 100, "min", 1))
batch_size =  int(st.number_input("Batch Size",  0, 1000, "min", 1))
max_iterations = int(st.number_input("Maximum number of iterations",  0, 1000, "min", 1))

if(st.button("Submit")):

    print("layer1 units: ", layer1_units)
    print("layer2 units: ", layer2_units)
    print("batch size: ", batch_size)
    print("max_iterations: ", max_iterations)
    
    accuracy_scores = ucla_neural_networks_model.main(layer1_units, layer2_units, batch_size, max_iterations)
    train_accuracy = accuracy_scores[0]
    pred_accuracy = accuracy_scores[1]

    st.write("Accuracy of model on training data")
    st.write(train_accuracy)

    st.write("Accuracy of model on prediction data")
    st.write(pred_accuracy)

    #add Streamlit code to display confusion matrix and loss curve
    st.write(
        """**Confusion Matrix**"""
    )
    st.image("./images/confusion_matrix.png")

    st.write(
    """**Loss Curve**"""
    )
    st.image("./images/loss_curve.png")


#Start Unsupervised Clustering Model Demo
# Set the page title and description
st.title("Unsupervised Clustering Model Demo")
st.write("""
This app demonstrates an unsupervised clustering model being used to cluster mall customers into distinct groups.
""")


unsupervised_clustering_model.main()

#Streamlit code to display images in HTML
st.write(
    """**Elbow Plot**"""
)
st.image("./images/elbow_plot.png")

st.write(
    """**Silhouette Plot**"""
)
st.image("./images/silhouette.png")

st.write(
    """**All Variables With K-Means Clustering**"""
)
st.image("./images/Variables3.png")
