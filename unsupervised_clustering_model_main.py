import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import streamlit as st 
import src.unsupervised_clustering_model as unsupervised_clustering_model


# Set the page title and description
st.title("Unsupervised Clustering Model Demo")
st.write("""
This app demonstrates an unsupervised clustering model being used to cluster mall customers into distinct groups.
""")
#Customer_ID Age	Annual_Income	Spending_Score

#annual_income = st.number_input("Annual Income")
#spending_score = st.number_input("Spending Score")


unsupervised_clustering_model.main()


#add Streamlit code to display images in HTML

st.write(
    """Elbow Plot"""
)
st.image("./images/elbow_plot.png")

st.write(
    """Silhouette"""
)
st.image("./images/silhouette.png")

st.write(
    """All Variables With K-Means Clustering"""
)
st.image("./images/Variables3.png")
