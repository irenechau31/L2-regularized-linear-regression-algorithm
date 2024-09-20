# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 16:41:07 2024

@author: User
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
train_data = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\Fordham\Data Mining (CISC-5790-L02)\HW1_dataset\train-1000-100.csv")

train_50_data=train_data.iloc[:50]
train_100_data=train_data.iloc[:100]
train_150_data=train_data.iloc[:150]

train_50_data.to_csv('train-50(1000)-100.csv', index=False)
train_100_data.to_csv('train-100(1000)-100.csv', index=False)
train_150_data.to_csv('train-150(1000)-100.csv', index=False)

# Split data into features X and target y
train_data = pd.read_csv('train-50(1000)-100.csv')
test_data =pd.read_csv(r"C:\Users\User\OneDrive\Desktop\Fordham\Data Mining (CISC-5790-L02)\HW1_dataset\test-1000-100.csv")

#using .values to convert pandas DataFrames to numpy arrays before performing matrix operations
x_train=train_data.iloc[:,:-1].values
y_train=train_data.iloc[:,-1].values

x_test=test_data.iloc[:,:-1].values
y_test=test_data.iloc[:,-1].values

# Ensure shapes match
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")
def L2_ridge_regression(x,y,lambd):
    n=x.shape[1] #shape[] refer to numbers of columns (1)/rows(0) in matrix x
    I=np.eye(n)
    # closed-form solution for L2
    w=np.linalg.inv(x.T@x+lambd*I)@x.T@y #one weight vector for the whole dataset of X features
    # use @ to multiply matrix
    print(f"w shape: {w.shape}")
    return w

def MSE(x,y,w):
        print(f"x shape: {x.shape}")
        print(f"w shape: {w.shape}")
        predictions = x @ w
        print(f"Predictions shape: {predictions.shape}")
        mse = np.mean((predictions - y) ** 2)
        return mse

lambdas_list=[]
mse_train_list=[]
mse_test_list=[]
for lambd in range (0,151):
    w=L2_ridge_regression(x_train,y_train,lambd)
    
    mse_train=MSE(x_train,y_train,w)
    mse_test=MSE(x_test,y_test,w)
    
    lambdas_list.append(lambd)
    mse_train_list.append(mse_train)
    mse_test_list.append(mse_test)
    
plt.plot(lambdas_list, mse_train_list, label='Training MSE', marker='o', color='blue')
plt.plot(lambdas_list, mse_test_list, label='Testing MSE', marker='x', color='red')
plt.xlabel('Lambda Values')
plt.ylabel('MSE')
plt.title('MSE vs. Lambda for Training and Testing Data')
plt.grid(True)
plt.legend()
plt.show()


    
    