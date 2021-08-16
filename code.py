import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('salary_data.csv')
X = data.iloc[:, :-1]
Y = data.iloc[:, 1]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train) 
table_train = plt
table_train.scatter(X_train,Y_train,color='red')
table_train.plot(X_train,regressor.predict(X_train),color='blue')
table_train.title('Salary V/S Experience(Train Data)')
table_train.xlabel('Salary')
table_train.ylabel('Years of Experience')
table_train.show()

table_test = plt;
table_test.scatter(X_test,Y_test,color='red')
table_test.plot(X_test,regressor.predict(X_test),color='blue')
table_test.title('Salary V/S Experience(Test Data)')
table_test.xlabel('Salary')
table_test.ylabel('Years of Experience')
table_test.show()
