'''
CST 2216
Term Project
David Xu (041173885)
real_estate_solution_model.py
Description: Modularization of real_estate_solution
'''


#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from .logger import exception_decorator
import pickle

'''
TO DO:
1. Only include fine-tuned Linear Regression model
2. App functions return calculated prices and MAE
'''

plt.gcf().set_dpi(300)

@exception_decorator
def data_prep(filepath):
    df = pd.read_csv(filepath)
    return df

#model before tuning
@exception_decorator
def LR_No_Tuning(df):
    # separate input features in x
    x = df.drop('price', axis=1)

    # store the target variable in y
    y = df['price']

    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, stratify=x.property_type_Bunglow)
    
    # train your model
    model = LinearRegression()
    lrmodel = model.fit(x_train, y_train)
    
    # make predictions on test set
    ypred = lrmodel.predict(x_test)

    #evaluate the model
    test_mae = mean_absolute_error(ypred, y_test)
    #print('Test error is', test_mae)


@exception_decorator
def DecisionTree(df):
    
    # separate input features in x
    x = df.drop('price', axis=1)

    # store the target variable in y
    y = df['price']

    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, stratify=x.property_type_Bunglow) #change stratify parameter in final copy
    
    # create an instance of the class
    dt = DecisionTreeRegressor(max_depth=3, max_features=10, random_state=567)
    # train the model
    dtmodel = dt.fit(x_train,y_train)
    
    # make predictions using the test set
    ytrain_pred = dtmodel.predict(x_train)

    # evaluate the model
    train_mae = mean_absolute_error(ytrain_pred, y_train)
    #print("train_mae: ", train_mae)

    # make predictions using the test set
    ytest_pred = dtmodel.predict(x_test)
    
    # evaluate the model
    test_mae = mean_absolute_error(ytest_pred, y_test)
    #print("test_mae: ", test_mae)

    # make predictions using the test set
    ytest_pred = dtmodel.predict(x_test)
    # evaluate the model
    test_mae = mean_absolute_error(ytest_pred, y_test)
    #print("mae on test set prediction (dt): ", test_mae)
    
    # make predictions on train set
    ytrain_pred = dtmodel.predict(x_train)
    # evaluate the model
    train_mae = mean_absolute_error(ytrain_pred, y_train)
    #print("mae on train set prediction (dt): ", test_mae)
    
    # Plot the tree with feature names
    tree.plot_tree(dtmodel, feature_names=dtmodel.feature_names_in_)

    tree.plot_tree(dtmodel)
    
    # Save the plot to a file
    plt.savefig('dt_tree.png')

    #pickle the model
    pickle.dump(dtmodel, open('./DT_Model','wb'))
    

@exception_decorator
def RandomForest(df):
    
    # separate input features in x
    x = df.drop('price', axis=1)

    # store the target variable in y
    y = df['price']

    #print variables
    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, stratify=x.property_type_Bunglow) #change stratify parameter in final copy
    
    #for individual tree attributes
    dt = DecisionTreeRegressor(max_depth=3, max_features=10, random_state=567)
    
    # train the model
    dtmodel = dt.fit(x_train,y_train)
    
    # create an instance of the model
    rf = RandomForestRegressor(n_estimators=200, criterion='absolute_error')
    
    # train the model
    rfmodel = rf.fit(x_train,y_train)
    
    # make prediction on train set
    ytrain_pred = rfmodel.predict(x_train)
    train_mae = mean_absolute_error(ytrain_pred, y_train)
    #print("mae on test set prediction (rf): ", train_mae)
    

    # make predictions on the x_test values
    ytest_pred = rfmodel.predict(x_test)
    test_mae = mean_absolute_error(ytest_pred, y_test)
    #print("mae on test set prediction (rf): ", test_mae)
    
    
    # evaluate the model
    test_mae = mean_absolute_error(ytest_pred, y_test)
    
    # Individual Decision Trees
    tree.plot_tree(rfmodel.estimators_[2], feature_names=dtmodel.feature_names_in_)
    tree.plot_tree(dtmodel)
    #plt.show()

    # Save the plot to a file
    plt.savefig('rf_tree.png')
    
    #pickle the model
    pickle.dump(rf, open('RF_Model','wb'))


#load model from pickle
@exception_decorator
def DecisionTreeApp(year_sold, property_tax, insurance, beds, baths, sqft, year_built, lot_size, basement, popular, recession, property_age, property_type_Bunglow, property_type_Condo):
    # Load the pickled model
    RE_Model = pickle.load(open('DT_Model','rb'))
    predicted_price = RE_Model.predict([[year_sold, property_tax, insurance, beds, baths, sqft, year_built, lot_size, basement,	popular, recession,	property_age, property_type_Bunglow, property_type_Condo]])
    return predicted_price
    
#load model from pickle
@exception_decorator
def RandomForestApp(year_sold, property_tax, insurance, beds, baths, sqft, year_built, lot_size, basement,	popular, recession, property_age, property_type_Bunglow, property_type_Condo):
    # Load the pickled model
    RE_Model = pickle.load(open('RF_Model','rb'))
    predicted_price = RE_Model.predict([[year_sold, property_tax, insurance, beds, baths, sqft, year_built, lot_size, basement, popular, recession, property_age, property_type_Bunglow, property_type_Condo]])

    return predicted_price


@exception_decorator
def main():     
    #initialize and train models
    df = data_prep('./data/final.csv')
    DecisionTree(df)
    RandomForest(df)  
    
if __name__ == '__main__':
    main()
    