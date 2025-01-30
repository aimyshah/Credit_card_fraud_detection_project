#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.linear_model import LogisticRegression


# In[2]:


credit_card_data = pd.read_csv('data(Credit_card_fraud_detection_project_#9)\credit_card_data.csv')


# In[3]:


# Prints fist five rows of the dataset
credit_card_data.head()


# In[4]:


# Basic info about the dataset.
credit_card_data.info()


# In[5]:


# Distribution of legit and fraudulant transactions.
credit_card_data['Class'].value_counts()


# In[6]:


# Inference: This is a highly imbalanced dataset
# 0 --> Normal transaction
# 1 --> Fraudulant transaction


# In[7]:


# Seperating the data for analysis.
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class ==1]


# In[8]:


print(legit)


# In[9]:


print(fraud)


# In[10]:


# Statistical measures of the data.
legit.Amount.describe()


# In[11]:


fraud.Amount.describe()


# In[12]:


# Comparing values of both transactions.
credit_card_data.groupby('Class').mean()


# ### Splitting the data into Features and Targets and handling imbalance using SMOTE

# In[13]:


X = credit_card_data.drop(columns='Class', axis=1)
Y = credit_card_data['Class']


# In[14]:


# Splitting into training and test data.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


# In[24]:


# Applying SMOTE
smote = SMOTE(sampling_strategy='auto')
X_resampled, Y_resampled = smote.fit_resample(X_train, Y_train)
print("Before SMOTE: ", Counter(Y))
print("After SMOTE: ", Counter(Y_resampled))


# In[16]:


# Plot the distribution before and after SMOTE.
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].bar(Counter(Y).keys(), Counter(Y).values(), color=['blue', 'red'])
ax[0].set_title("Before SMOTE")
ax[0].set_xticks([0, 1])
ax[0].set_xlabel("Class")
ax[0].set_ylabel("Count")

ax[1].bar(Counter(Y_resampled).keys(), Counter(Y_resampled).values(), color=['blue', 'red'])
ax[1].set_title("After SMOTE")
ax[1].set_xticks([0, 1])
ax[1].set_xlabel("Class")
ax[1].set_ylabel("Count")

plt.show()


# ### Model Training

# #### Logistic Regression

# In[17]:


model = LogisticRegression()


# In[18]:


# Training the LogisticRegression with the training data.
model.fit(X_resampled, Y_resampled)


# ### Model Evaluation

# In[19]:


# Accuracy on training data.
training_data_predictions = model.predict(X_train)
training_data_accuracy = accuracy_score(training_data_predictions, Y_train)
print("Accuracy on training data: ", round(training_data_accuracy * 100, 2))


# In[20]:


# Accuracy on testing data.
test_data_predictions = model.predict(X_test)
test_data_accuracy = accuracy_score(test_data_predictions, Y_test)
print("Accuracy on testing data: ", round(test_data_accuracy * 100, 2))


# ### Making a Predictive System

# In[25]:


def predict_transaction(time, amount, features):
    # Convert into a numpy array
    input_data = np.array([time] + features + [amount]).reshape(1, -1)

    # Predict
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    """ Here [0][1] is the index of probability of fraud from 2D array returned
    by the predict_proba function where the first value is the probability of 
    transactions being legit. e.g. [0.85, 0.15] is returned by the function. """

    return "Fraud" if prediction[0] == 1 else "Legit", probability


# In[26]:


# Testing with sample transaction.
sample_time = 472
sample_amount = 529
sample_features = [-3.0435406239976,-3.15730712090228,1.08846277997285,2.2886436183814,1.35980512966107,-1.06482252298131,0.325574266158614,-0.0677936531906277,-0.270952836226548,-0.838586564582682,-0.414575448285725,-0.503140859566824,0.676501544635863,-1.69202893305906,2.00063483909015,0.666779695901966,0.599717413841732,1.72532100745514,0.283344830149495,2.10233879259444,0.661695924845707,0.435477208966341,1.37596574254306,-0.293803152734021,0.279798031841214,-0.145361714815161,-0.252773122530705,0.0357642251788156]

result, prob = predict_transaction(sample_time, sample_amount, sample_features)
print(f"Prediction: {result}, Probability of Fraud: {prob: .2f}")


# In[ ]:




