# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 23:24:37 2019

@author: Yousra
"""

# This is Kaggle Machine learning course 1
# Use data of houses to predict the price of a houses using Decision Trees 
# The dataset is called IWOA

#Step One : Read the dataset file and show the data with the mean and min and max and ..... etc.
import pandas as pd
data_path = 'C:\Users\Yousra\Desktop\Masters\MachineLearning\Course\Kaggle\MachineLearning1\melbData.csv'
melb_data= pd.read_csv(data_path)
melb_data.describe()
z = melb_data.iloc[:, :-1].values
# Step Two : chose features for your model 

# Here we print the columns names to select from them which features and to execlude the target feature
melb_data.columns
# Here we remove the NaN data lines
melb_data = melb_data.dropna(axis=0)

Ans = melb_data.Price
features =['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
Input = melb_data[features]
Input.describe()
Input.head() # few first rows
# Step3: prepare your model 
# 4 steps : prepare - fit - predict - evaluate 
# Prepare the model: choose the model and which parameters 
from sklearn.tree import DecisionTreeRegressor
# Define model. Specify a number for random_state to ensure same results each run
melb_model = DecisionTreeRegressor(random_state=1)
# Fit model
melb_model.fit(Input, Ans)
# Predictions
print("Making predictions for the following 5 houses:")
print(Input.head())
print("The predictions are")
print(melb_model.predict(Input.head()))

#Now evaluating the model by measuring the accuracy or predicting results 
#using the mean abolute error
from sklearn.metrics import mean_absolute_error
predicted_prices = melb_model.predict(Input)
error = mean_absolute_error(Ans,predicted_prices )
print(error)

#lets make things better (train and test on different datasets)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
train_Input, test_Input, train_Ans, test_Ans = train_test_split(Input, Ans, random_state=0)

melb_model_advanced= melb_model = DecisionTreeRegressor(random_state=1)
train_Input.describe()
train_Ans.describe()
melb_model_advanced.fit(train_Input, train_Ans)

predicted_prices = melb_model_advanced.predict(test_Input)
error = mean_absolute_error(test_Ans,predicted_prices )
print(error)

# Now we cosider the over/underfitting 

from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max, train_Input, test_Input, train_Ans, test_Ans):
    
    model =   DecisionTreeRegressor (max_leaf_nodes=max, random_state=0)
    model.fit(train_Input, train_Ans)
    predicted_vals= model.predict(test_Input)
    mae= mean_absolute_error(predicted_vals,test_Ans)
    return(mae)
 
for max in [5,50,500,5000]:
     my_mae = get_mae(max,train_Input, test_Input, train_Ans, test_Ans )
     print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max, my_mae))
#After checking the results we find that the 500 leaves are the optimal number of leaves
     
     
# Now checking another Algorithm calles Random Forrests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_Input, train_Ans)
forest_model_preds = forest_model.predict(test_Input)
print(mean_absolute_error(test_Ans,forest_model_preds))   #Further Improvements could be applied but it's better than the optimal value of tree leaves in the tree regression model
# Done here








