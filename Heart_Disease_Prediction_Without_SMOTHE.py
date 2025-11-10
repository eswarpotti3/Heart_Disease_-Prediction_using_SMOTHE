# %%
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
y= data['DEATH_EVENT']
x=data.drop(['DEATH_EVENT'],axis=1)

# %%
#importing train test split to create validation set
from sklearn.model_selection import train_test_split

# %%
#creating the train and validation set
X_train, X_valid, y_train, y_valid = train_test_split(x, y, random_state = 123, stratify=y, test_size=0.30)

# %%
type(y_train)

# %%
#distribution in traing set
y_train.value_counts(normalize=True)

# %%
#distribution in validation set
y_valid.value_counts(normalize=True)

# %%
#shape of training set
X_train.shape, y_train.shape

# %%
#shape of validation set
X_valid.shape, y_valid.shape

# %%
from sklearn.ensemble import ExtraTreesClassifier

# %%
etc_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
etc_clf = etc_clf.fit(X_train, y_train)

# %% [markdown]
# Extra Tree Classifier()

# %%
#Predict class for X
pred_etc = etc_clf.predict(X_valid)

# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# %%
etc_acc=accuracy_score(y_valid,pred_etc)
print(f'Accuracy is {(etc_acc)}')
etc_prec= precision_score(y_valid,pred_etc)
print(f"The precision is {(etc_prec)}")
etc_rec= recall_score(y_valid,pred_etc)
print(f"The recall is {(etc_rec)}")

etc_f1=f1_score(y_valid,pred_etc)
print(f"The f1 score is {(etc_f1)}")


