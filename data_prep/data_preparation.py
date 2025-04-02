#rename this file

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def credit_eligibility_data_prep(df):
    # convert columns to object type
    df['Credit_History'] = df['Credit_History'].astype('object')
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].astype('object')
    
    # impute all missing values in all the features
    #Categorical variables
    df['Gender'].fillna('Male', inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

    #Numerical variable
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    
    # drop 'Loan_ID' variable from the data. We won't need it.
    df = df.drop('Loan_ID', axis=1)
    
    # replace values in Loan_approved column
    df['Loan_Approved'] = df['Loan_Approved'].replace({'Y':1, 'N':0})
    
    # Create dummy variables for all 'object' type variables except 'Loan_Status'
    df = pd.get_dummies(df, drop_first=True)   

    #return the prepared dataframe
    return df

def neural_networks_data_prep(df):
    '''data preparation for ucla_neural_networks_model'''
    
    # Converting the target variable into a categorical variable
    df['Admit_Chance']=(df['Admit_Chance'] >=0.8).astype(int)
    # Dropping unnecessary columns
    df = df.drop(['Serial_No'], axis=1)
    # convert University_Rating to categorical type
    df['University_Rating'] = df['University_Rating'].astype('object')
    df['Research'] = df['Research'].astype('object')
    # Create dummy variables for all 'object' type variables except 'Loan_Status'
    cleaned_df = pd.get_dummies(df, columns=['University_Rating','Research'],dtype='int')
    
    return cleaned_df