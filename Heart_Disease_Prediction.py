# %%
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
#reading
data=pd.read_csv('heart_failure_clinical_records_dataset.csv')

# %%
#shape of the data
data.shape

# %%
#first five rows of the data
data.head()

# %%
#checking for the missing value in the data
data.isnull().sum()

# %% [markdown]
# FS

# %%
#droping unimportant variables separating independent and dependent variables
datasf = data.drop(['anaemia', 'high_blood_pressure', 'diabetes', 'smoking'], axis=1)
labels = datasf['DEATH_EVENT']
features = datasf.drop('DEATH_EVENT', axis=1)

# %%
#importing train test split to create validation set
from sklearn.model_selection import train_test_split

# %%
 # perform train test split
train_sifeatures, test_sifeatures, train_silabels, test_silabels = train_test_split(features, labels, test_size=0.30 ,random_state=123)

# %% [markdown]
# SMOTHE

# %%
from imblearn.over_sampling import SMOTE
sm = SMOTE()

# %%
train_features1, train_labels1 = sm.fit_resample(train_sifeatures, train_silabels)
test_features1, test_labels1 = sm.fit_resample(test_sifeatures, test_silabels)

# %%
type(train_silabels)

# %%
#distribution in traing set
train_silabels.value_counts(normalize=True)

# %%
#distribution in validation set
test_silabels.value_counts(normalize=True)

# %%
#shape of training set
train_sifeatures.shape, train_silabels.shape

# %%
#shape of validation set
test_sifeatures.shape, test_silabels.shape

# %% [markdown]
# Extra Tree Classifier

# %%
from sklearn.ensemble import ExtraTreesClassifier

# %%
# train the model
etc_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
etc_clf = etc_clf.fit(train_features1,train_labels1) #Build a decision tree classifier from the training set

# %%
pred_etc = etc_clf.predict(test_features1) #Predict class for X.

# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# %%
etc_acc=accuracy_score(test_labels1,pred_etc)
print(f'Accuracy is {(etc_acc)}')

etc_prec= precision_score(test_labels1,pred_etc)
print(f"The precision is {(etc_prec)}")

etc_rec= recall_score(test_labels1,pred_etc)
print(f"The recall is {(etc_rec)}")

etc_f1=f1_score(test_labels1,pred_etc)
print(f"The f1 score is {(etc_f1)}")


