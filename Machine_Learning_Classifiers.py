import pandas as pd
import pyarabic.araby
from Preprocessing import doPreprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


# Classifiers:   SVM ,  NB ,  LR , KNN , DT , RF
# Feature extraction: N-Gram , TF-IDF

# Training function
def training(classifier, featured_comment, y):
    X_train, X_test, y_train, y_test = train_test_split(featured_comment, y,test_size=0.30)  # Split data into 70% training and 30% testing


    if classifier == "SVM":
        svm(X_train, y_train, X_test, y_test)

    elif classifier == "NB":
        nb(X_train, y_train, X_test, y_test)

    elif classifier == "LR":
        lr(X_train, y_train, X_test, y_test)

    elif classifier == "KNN":
        knn(X_train, y_train, X_test, y_test)

    elif classifier == "DT":
        dt(X_train, y_train, X_test, y_test)

    elif classifier == "rf":
        rf(X_train, y_train, X_test, y_test)

    else:
        print("Wrong classifier input")


# SVM function
def svm(X_train, y_train, X_test, y_test):
    # # GridSearchCV to find best tune
    # param_grid={"C":[0.1,1,100,1000],"kernel":["linear","rbf","poly","sigmoid"],'degree':[1,2,3,4,5,6]}
    # grid = GridSearchCV(SVC(),param_grid,n_jobs=-1)
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

# Naive Bayes function
def nb(X_train, y_train, X_test, y_test):
    pass # Remove pass when you start coding on this classifier

# Logistic Regression function
def lr(X_train, y_train, X_test, y_test):
    pass # Remove pass when you start coding on this classifier

# K-Nearest Neighbors function
def knn(X_train, y_train, X_test, y_test):
    pass # Remove pass when you start coding on this classifier

# Decision Tree function
from sklearn import tree
def dt(X_train, y_train, X_test, y_test):
    # Building the model
    dtClassf = tree.DecisionTreeClassifier() # create the classifier DT

    # Training the model
    print("Training the model....\n")
    dtClassf.fit(X_train,y_train) # train the classifier

    # evaluate the model and printing the result
    y_pred = dtClassf.predict(X_test)
    accuracyT = metrics.accuracy_score(y_test, y_pred)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print("The accuracy score of DT: ",accuracyT)
    print("-----------------------------------------")

# Random Forest function
def rf(X_train, y_train, X_test, y_test):

# Create an instance of the RandomForestClassifier
    # # For tuning and finding the best parameters and score for Rain Forest classifier
    # rfClassf = RandomForestClassifier()
    # param_grid = {
    # 'n_estimators': [10, 50, 100, 200],
    # 'max_depth': [None, 5, 10, 20],
    # 'min_samples_split': [2, 5, 10],
    # 'min_samples_leaf': [1, 2, 4]
    # }
    # grid_search = GridSearchCV(estimator=rfClassf, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)

    # # Fit the GridSearchCV object to the training data
    # grid_search.fit(X_train, y_train)

    # # Print the best parameters and the best score
    # print("Best parameters: ", grid_search.best_params_)
    # print("Best score: ", grid_search.best_score_)

    # Building the model
    rfClassf = RandomForestClassifier(max_depth= None, min_samples_leaf= 2, min_samples_split= 4, n_estimators= 100)

    # Training the model
    print("Training the model....\n")
    rfClassf.fit(X_train,y_train) # train the classifier

    # evaluate the model and printing the result
    y_pred = rfClassf.predict(X_test)
    accuracyT = metrics.accuracy_score(y_test, y_pred)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print("The accuracy score of RainForest: ",accuracyT)
    print("-----------------------------------------")


# preparing the dataset
def prepare():
    # Read CSV file(Dataset)
    df = pd.read_csv("finalDataset.csv", encoding="utf-8-sig") # read dataset

    # Encode comment and preparing X
    X = df["Comment"].apply(doPreprocessing)

    # converting labels to numbers and preparing y
    le = preprocessing.LabelEncoder()
    le.fit(df.Label)
    df["Label"] = le.transform(df.Label)
    y = df["Label"]
    return X,y


# TF-IDF feature extraction function
def tfidf(classifier):
    X, y = prepare() # call prepare function
    tfidf_vectorizer = TfidfVectorizer(tokenizer=pyarabic.araby.tokenize, use_idf=True)
    featured_comment = tfidf_vectorizer.fit_transform(X)
    training(classifier, featured_comment, y) # call training function

# N-Grams feature extraction function
def n_gram(classifier, n):
    X, y = prepare() # call prepare function
    vectorizer = CountVectorizer(ngram_range=(n,n))
    featured_comment = vectorizer.fit_transform(X)
    training(classifier, featured_comment, y) # call training function

# This function to start the process of training
def start(classifier, feature_extraction, n):
    if feature_extraction == "ngram":
        n_gram(classifier,n)
    elif feature_extraction == "tfidf":
        tfidf(classifier)
    else:
        print("Wrong feature extraction input")


# First parameter choose one of these classifiers: SVM, NB, LR, KNN, DT, RF
# Second parameter choose one of these feature extraction: ngram, tfidf
# Third parameter enter n-gram level: 1, 2, 3 (this parameter will be ignored if TF-IDF choosen, but must enter any number)
start("rf","ngram",1)