# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn. Apply new unknown values

## Program:
```

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: T.Roshini
RegisterNumber: 2122232320175

import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:, : -1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print("Y_pred:",y_pred)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("accuracy:",accuracy)
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print("confusion:",confusion)
```

## Output:

#### Head:
![Screenshot 2024-10-20 151714](https://github.com/user-attachments/assets/682a260d-2c89-4648-953f-27c7c1b9678b)

#### data1.head:
![Screenshot 2024-10-20 151759](https://github.com/user-attachments/assets/16fde5b9-44c0-4087-aa4f-2e0b6ff6f2b9)

#### Sum null:
![Screenshot 2024-10-20 151831](https://github.com/user-attachments/assets/b736a7d8-ba76-4f1d-848c-a393b40d47f1)

#### Sum duplicate:
![Screenshot 2024-10-20 151852](https://github.com/user-attachments/assets/29492f10-ba1d-426a-807a-57950460c598)

#### Data1:
![Screenshot 2024-10-20 151927](https://github.com/user-attachments/assets/bfa9bcac-7a55-4910-b42d-8a350f914732)

#### X:
![Screenshot 2024-10-20 152000](https://github.com/user-attachments/assets/7542c680-4f3d-416f-8741-5e30e9bded97)

#### Y:
![Screenshot 2024-10-20 152026](https://github.com/user-attachments/assets/d04582a0-1ca8-45fa-b332-ee8bb1cd685b)

#### LogisticRegression:
![Screenshot 2024-10-20 152103](https://github.com/user-attachments/assets/6b27ff9a-bd59-4a98-9341-7e2b59edd1b0)

#### Y_Predict:
![Screenshot 2024-10-20 152130](https://github.com/user-attachments/assets/a1a7dad0-bc6e-43f1-9e2a-3578ffa57cc5)

#### Accuracy:
![Screenshot 2024-10-20 152156](https://github.com/user-attachments/assets/e5758414-df53-4c5a-aeda-99467bf7831f)

#### Confusion matrix:
![Screenshot 2024-10-20 152222](https://github.com/user-attachments/assets/c041df48-ed4f-4685-94ed-b275c10e89f7)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.



