# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the required packages and print the present data
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Muhammad Afshan A
RegisterNumber:  212223100035
*/

import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()
```
![alt text](<ml ex-5(1).png>)

```

data1=data.copy()
data1=data1.drop(["sl_no","salary"], axis=1)
data1.head()
```
![alt text](<ml ex-5(2).png>)

```
data1.isnull().sum()
```

![alt text](<ml ex-5(3).png>)

```
data1.duplicated().sum()
```

![alt text](<ml ex-5(4).png>)

```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1 ["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
```

![alt text](<ml ex-5(5).png>)

```
x=data1.iloc[:,:-1]
x
```

![alt text](<ml ex-5(6).png>)

```py
y=data1["status"]
y
```
![alt text](<ml ex-5(7).png>)

```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression (solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
```
![alt text](<ml ex-5(8).png>)

```
from sklearn.metrics import accuracy_score
accuracy= accuracy_score(y_test,y_pred) 
accuracy
```
![alt text](<ml ex-5(9).png>)

```
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
```
![alt text](<ml ex-5(10).png>)

```
x_new=pd.DataFrame([[1,80,1,90,1,1,90,1,0,85,1,85]],columns=['gender', 'ssc_p', 'ssc_b', 'hsc_p', 'hsc_b', 'hsc_s','degree_p', 'degree_t', 'workex', 'etest_p', 'specialisation', 'mba_p'])
print('Name: Muhammad Afshan A')
print('Reg No: 212223100035')
lr.predict(x_new)
```

![alt text](<ml ex-5(11).png>)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
