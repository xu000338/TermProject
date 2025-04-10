import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 

import sys
sys.path.insert(0, 'C:\\Users\\davex\\Documents\\College\\CST2216\\Project\\data_prep')
from data_prep.data_preparation import neural_networks_data_prep

import warnings
warnings.filterwarnings("ignore")

#import logger module
from .logger import exception_decorator

@exception_decorator
def load(filename):
    
    # load the data using the pandas `read_csv()` function.
    data = pd.read_csv(filename)
 
    data = neural_networks_data_prep(data)

    x = data.drop(['Admit_Chance'], axis=1)
    y = data['Admit_Chance']
    
    return x, y

@exception_decorator
def split(x, y):
    # split the data
    from sklearn.model_selection import train_test_split
    # Splitting the dataset into train and test data
    xtrain, xtest, ytrain, ytest =  train_test_split(x, y, test_size=0.2, random_state=123, stratify=y)
    
    return xtrain, xtest, ytrain, ytest

@exception_decorator
def transform(xtrain, xtest):
  
    # import standard scaler
    from sklearn.preprocessing import MinMaxScaler

    # fit calculates the mean and standard deviation
    scaler = MinMaxScaler()
    scaler.fit(xtrain)
    # Now transform xtrain and xtest
    xtrain_scaled = scaler.transform(xtrain)
    xtest_scaled = scaler.transform(xtest)

    xtrain_scaled_df = pd.DataFrame(xtrain_scaled, columns=xtrain.columns)
    
    return xtrain_scaled, xtest_scaled
  

@exception_decorator
def train(ytrain, ytest, xtrain_scaled, xtest_scaled, layer1_units, layer2_units, custom_batch_size, max_iterations):

    # import the model
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
    # fit/train the model. Check batch size.
    MLP = MLPClassifier(hidden_layer_sizes=(layer1_units, layer2_units), batch_size=custom_batch_size, max_iter=max_iterations, random_state=123)
    MLP.fit(xtrain_scaled,ytrain)

    # make predictions on train
    ypred_train = MLP.predict(xtrain_scaled)

    # check accuracy of the model on training
    train_accuracy = accuracy_score(ytrain, ypred_train)

    # make Predictions
    ypred = MLP.predict(xtest_scaled)

    # check accuracy of the model on prediction
    pred_accuracy = accuracy_score(ytest, ypred)


    cm = confusion_matrix(ytest, ypred)# Plotting loss curve
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot().figure_.savefig('./images/confusion_matrix.png')
    loss_values = MLP.loss_curve_

    # Plotting the loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label='Loss', color='blue')
    plt.title('Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('./images/loss_curve.png')
    
    #save model to pickle file
    pickle.dump(MLP, open('MLP_Model', 'wb'))
    
    return train_accuracy, pred_accuracy

def modelApp(gre_score,	toefl_score, university_rating,	sop, lor, cgpa,	research):
    model = pickle.load(open('MLP_Model', 'rb'))
    prediction = model.predict([[gre_score,	toefl_score, university_rating,	sop, lor, cgpa,	research]])
    return prediction

def main(layer1_units, layer2_units, batch_size, max_iterations):
    
    x, y = load('./data/Admission.csv')
    xtrain, xtest, ytrain, ytest = split(x, y)
    xtrain_scaled, xtest_scaled = transform(xtrain, xtest)
    accuracy_scores = train(ytrain, ytest, xtrain_scaled, xtest_scaled, layer1_units, layer2_units, batch_size, max_iterations)
    return accuracy_scores
