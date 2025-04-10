# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 11:09:21 2025

@author: davex
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# Import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

# Import the data_prep function from the data_preparation.py file
from data_prep.data_preparation import credit_eligibility_data_prep
#from data_prep import data_prep

# For splitting the data
from sklearn.model_selection import train_test_split


import functools
from functools import reduce

from .logger import exception_decorator

#main script
@exception_decorator
def main():
    
    # Import the data from 'credit.csv'
    df = pd.read_csv('./data/credit.csv')
    
    # Clean and Prepare the data
    df = credit_eligibility_data_prep(df)
    
    # Seperate the input features and target variable
    x = df.drop('Loan_Approved',axis=1)
    y = df.Loan_Approved
    
    # splitting the data in training and testing set
    xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.2, stratify=y)
  
    #hyperparameter tuning
    rfmodel = RandomForestClassifier(n_estimators=2, max_depth=2, max_features=10)  #for testing only
    rfmodel.fit(xtrain, ytrain)
    
    # SAVE THE MODEL
    #create a file named random_forest_credit.pickle in write-binary mode
    rf_pickle = open("./data/random_forest_credit.pickle", "wb")

    #serialize the model object rfmodel and write it to the file rf_pickle.
    pickle.dump(rfmodel, rf_pickle)
    rf_pickle.close()

    fig, ax = plt.subplots()

    ax = sns.barplot(x=rfmodel.feature_importances_, y=x.columns)
    plt.title("Feature importance chart")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    fig.savefig("./images/feature_importance.png")

if __name__ == "__main__":
    main()
 