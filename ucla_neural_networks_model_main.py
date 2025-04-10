import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import streamlit as st 
from src.logger import exception_decorator

import src.ucla_neural_networks_model as ucla_neural_networks_model

def main():
    # Set the page title and description
    st.title("UCLA Neural Networks Model Demo")
    st.write("""
    This app demonstrates a neural network model being used to predict the probability of admission to UCLA based on GRE and TOEFL scores.
    """)
    
    # modelApp(gre_score, toefl_score, university_rating, sop, lor, cgpa, research)
    
    
    #gre_score = st.number_input("GRE Score")
    #toefl_score = st.number_input("TOEFL Score")
    #university_ranking = st.selectbox("University Ranking", [0, 1, 2, 3, 4, 5])
    #sop = st.selectbox("SOP", [0, 1, 2, 3, 4])
    #lor = st.selectbox("LOR", [0, 1, 2, 3, 4, 5])
    #cgpa = st.number_input("CGPA")
    #research = st.selectbox("Research", [0, 1])
    
    #if(st.button("Predict Eligibility")):
    #    eligibilty = ucla_neural_networks_model.modelApp(gre_score, toefl_score, university_ranking, sop, lor, cgpa, research)    
    #    st.write("Your probability of acceptance to UCLA graduate program")
    #    st.write(eligibilty)   
    
    ucla_neural_networks_model.main()

    #add Streamlit code to display confusion matrix and loss curve
    st.write(
        """Confusion Matrix"""
    )
    st.image("./images/confusion_matrix.png")

    st.write(
    """Loss Curve"""
    )
    st.image("./images/loss_curve.png")

if __name__ == '__main__':
    main()