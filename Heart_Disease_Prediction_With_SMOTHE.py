#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
#reading
data=pd.read_csv('heart_failure_clinical_records_dataset.csv') # Assuming the file is in the same directory as the notebook

# %%
#shape of the data
data.shape

# %%
#first five rows of the data
data.head()

# %%
#checking for the missing value in the data
data.isnull().sum()

# %%
#separating independent and dependent variables
labels= data['DEATH_EVENT']
features=data.drop(['DEATH_EVENT'],axis=1)

# %%
#importing train test split to create validation set
from sklearn.model_selection import train_test_split

# %%
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.30, random_state=123)


# %% [markdown]
# SMOTHE

# %%
from imblearn.over_sampling import SMOTE

# %%
sm = SMOTE()
Strain_features, Strain_labels = sm.fit_resample(train_features, train_labels)
Stest_features, Stest_labels = sm.fit_resample(test_features, test_labels)

# %%
type(train_labels)

# %%
#distribution in traing set
train_labels.value_counts(normalize=True)

# %%
#distribution in validation set
test_labels.value_counts(normalize=True)

# %%
#shape of training set
train_features.shape, train_labels.shape

# %%
#shape of validation set
test_features.shape, test_labels.shape

# %%
from sklearn.ensemble import ExtraTreesClassifier

# %%
# train the model
etc_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
etc_clf = etc_clf.fit(Strain_features,Strain_labels) #Build a decision tree classifier from the training set

# %% [markdown]
# Extra Tree Classifier

# %%
#Predict class for X.
pred_etc = etc_clf.predict(Stest_features)

# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# %%
etc_acc=accuracy_score(Stest_labels,pred_etc)
print(f'Accuracy is {(etc_acc)}')

etc_prec= precision_score(Stest_labels,pred_etc)
print(f"The precision is {(etc_prec)}")

etc_rec= recall_score(Stest_labels,pred_etc)
print(f"The recall is {(etc_rec)}")

etc_f1=f1_score(Stest_labels,pred_etc)
print(f"The f1 score is {(etc_f1)}")



