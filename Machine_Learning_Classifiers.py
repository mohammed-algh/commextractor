import pandas as pd
from Preprocessing import doPreprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn import preprocessing
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


# Classifiers:   SVM ,  NB ,  LR , KNN , DT , RF
# Feature extraction: N-Gram , TF-IDF


# Read CSV file(Dataset)
df = pd.read_csv("finalDataset.csv", encoding="utf-8-sig") # read dataset

# Encode comment and preparing X
X = df["Comment"].apply(doPreprocessing)
cv = CountVectorizer()
X = cv.fit_transform(X).toarray()

# converting labels to numbers and preparing y
le = preprocessing.LabelEncoder()
le.fit(df.Label)
df["Label"] = le.transform(df.Label)
y = df["Label"]



#Preparing the dataset
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=80) # Split data into 70% training and 30% testing

# # GridSearchCV to find best tune
# param_grid={"C":[0.1,1,100,1000],"kernel":["linear","rbf","poly","sigmoid"],'degree':[1,2,3,4,5,6]}
# grid = GridSearchCV(SVC(),param_grid)
# grid.fit(X_train,y_train)
# print(grid.best_params_)
# print(grid.score(X_test,y_test))


# Building the model
svmClassf = SVC(kernel="linear",C=1,degree=1) # create the classifier SVM

# Training the model
print("Training the model....\n")
svmClassf.fit(X_train,y_train) # train the classifier

# evaluate the model and printing the result
y_pred = svmClassf.predict(X_test)
accuracyT = metrics.accuracy_score(y_test, y_pred)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("The accuracy score of SVM: ",accuracyT)
print("-----------------------------------------")