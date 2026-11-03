import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#Data Collection and Analysis

diabetes_dataset = pd.read_csv('I:\\Diabetes prediction\\diabetes.csv')

# print(diabetes_dataset.head())
# print(diabetes_dataset.tail())
# print(diabetes_dataset.describe())

print(diabetes_dataset['Outcome'].value_counts())

# Diabetic patient value: 1
# Non-diabetic patient value: 0

print(diabetes_dataset.groupby('Outcome').mean())

# seperating the data and labels by dropping and creating variables

x= diabetes_dataset.drop(columns= 'Outcome', axis=1)
y= (diabetes_dataset['Outcome'])

print(x,y)

# Data Standarization 
#creating variables:

scaler = StandardScaler()
stnadarized_data= scaler.fit_transform(x)

print(stnadarized_data)

#again creating variables and fitting standarized data into 'x' variable

x = stnadarized_data
y = diabetes_dataset['Outcome']

print(x,y)

# Train_test_split method 

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, stratify=y, random_state=2)
print(x.shape, x_train.shape, x_test.shape)

# Training the model 

classifier = svm.SVC(kernel= 'linear')

#Training the support vector machine classifier 

classifier.fit(x_train,y_train)


# Model Evaluation 
#Accuracy score of training data:

x_train_prediction = classifier.predict(x_train) 
training_data_accuracy= accuracy_score(x_train_prediction, y_train)

print('accuracy score of trained data:', training_data_accuracy)


#Accuracy score of testing data:

x_test_prediction = classifier.predict(x_test)
test_data_accuracy= accuracy_score(x_test_prediction, y_test)

print('accuracy score of test data:', test_data_accuracy)


# Making a predictive system 

input_data = (2,174,88,37,120,44.5,0.646,24)

#converting to numpy array to make it efficient 

input_data_as_np_array = np.asarray(input_data)

#Reshaping the array as we are predicting one array not all 

input_data_reshaped = input_data_as_np_array.reshape(1,-1)

#standarized the input data 

std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

# if else 

if (prediction[0] == 0):
    print('The person is Non-diabetic')

else:
    print('The person is diabetic') 


## Saving the Trained Model

import pickle

filename = 'diabetes_model.sav'
pickle.dump(classifier, open(filename,'wb'))

#loading the trained model

loaded_model = pickle.load(open('diabetes_model.sav', 'rb'))

#Loading the saved data

input_data = (2,174,88,37,120,44.5,0.646,24)

#converting to numpy array to make it efficient 

input_data_as_np_array = np.asarray(input_data)

#Reshaping the array as we are predicting one array not all 

input_data_reshaped = input_data_as_np_array.reshape(1,-1)

prediction = classifier.predict(input_data_reshaped)
print(prediction)


