import pandas as pd
import pyarabic.araby
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from Preprocessing import doPreprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from time import time


# Classifiers:   SVM ,  NB ,  LR , KNN , DT , RF
# Feature extraction: N-Gram , TF-IDF

def calculating(classifier, param_grid, X_train, X_test, y_train, y_test):
    estimators = Pipeline([
        ("vectorizer", CountVectorizer(preprocessor= doPreprocessing)),
        ("transformer", TfidfTransformer()),
        ("classifier", classifier)
    ])
    gsc = GridSearchCV(estimators, param_grid, cv=5, n_jobs=-1, verbose=True) # change n_jobs to (your cpu cores - 2) to make your device more comfort

    start_time = time()
    print("Training the model...")
    gsc.fit(X_train, y_train)
    print("Training finished", time() - start_time," Seconds")
    y_pred = gsc.predict(X_test)
    best_param = gsc.best_params_
    best_score = gsc.best_score_

    print_result(y_test, y_pred, best_score, best_param)
    for param_name in sorted(param_grid.keys()):
        print("%s: %r" % (param_name, gsc.best_params_[param_name]))


# Training function
def training(classifier, featured_comment, y, pip_status):
    X_train, X_test, y_train, y_test = train_test_split(featured_comment, y,test_size=0.30)  # Split data into 70% training and 30% testing


    if classifier == "SVM":
        svm(X_train, y_train, X_test, y_test, pip_status)

    elif classifier == "NB":
        nb(X_train, y_train, X_test, y_test, pip_status)

    elif classifier == "LR":
        lr(X_train, y_train, X_test, y_test, pip_status)

    elif classifier == "KNN":
        knn(X_train, y_train, X_test, y_test, pip_status)

    elif classifier == "DT":
        dt(X_train, y_train, X_test, y_test, pip_status)

    elif classifier == "RF":
        rf(X_train, y_train, X_test, y_test, pip_status)

    else:
        print("Wrong classifier input")


# SVM function
def svm(X_train, y_train, X_test, y_test, pip_status):
    # # GridSearchCV to find best tune
    # param_grid={"C":[0.1,1,100,1000],"kernel":["linear","rbf","poly","sigmoid"],'degree':[1,2,3,4,5,6]}
    # grid = GridSearchCV(SVC(),param_grid,n_jobs=-1)
    # grid.fit(X_train,y_train)
    # print(grid.best_params_)
    # print(grid.score(X_test,y_test))

    classifier = SVC() # Change SVC() with your classifier
    if pip_status:
        param_grid = {
            'vectorizer__ngram_range': [(1, 1)], #(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3) TRY EACH ONE
            'transformer__use_idf': [False, True],
            'transformer__norm': ['l1','l2'],
            'classifier__kernel': ['linear','rbf'],
            'classifier__C': [0.1,1,100],
            'classifier__gamma':[0.1,1,'auto'],
            'classifier__decision_function_shape': ['ovr','ovo']
            # add yours parameters here (Note: you must write it like this: 'classifier__{name of the parameter}': [ ]
        }

        calculating(classifier, param_grid, X_train, X_test, y_train, y_test)

    else:

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
def nb(X_train, y_train, X_test, y_test, pip_status):

    classifier = SVC()  # Change SVC() with your classifier
    if pip_status:
        param_grid = {
            'vectorizer__ngram_range': [(1, 1)], #(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3) TRY EACH ONE
            'transformer__use_idf': [False, True],
            'transformer__norm': ['l1', 'l2'],
            # add yours parameters here (Note: you must write it like this: 'classifier__{name of the parameter}': [ ]
        }

        calculating(classifier, param_grid, X_train, X_test, y_train, y_test)

    else:
        pass


# Logistic Regression function
def lr(X_train, y_train, X_test, y_test, pip_status):

    classifier = SVC()  # Change SVC() with your classifier
    if pip_status:
        param_grid = {
            'vectorizer__ngram_range': [(1, 1)], #(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3) TRY EACH ONE
            'transformer__use_idf': [False, True],
            'transformer__norm': ['l1', 'l2'],
            # add yours parameters here (Note: you must write it like this: 'classifier__{name of the parameter}': [ ]
        }

        calculating(classifier, param_grid, X_train, X_test, y_train, y_test)

    else:
        pass


# K-Nearest Neighbors function
def knn(X_train, y_train, X_test, y_test, pip_status):

    classifier = SVC()  # Change SVC() with your classifier
    if pip_status:
        param_grid = {
            'vectorizer__ngram_range': [(1, 1)], #(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3) TRY EACH ONE
            'transformer__use_idf': [False, True],
            'transformer__norm': ['l1', 'l2'],
            # add yours parameters here (Note: you must write it like this: 'classifier__{name of the parameter}': [ ]
        }

        calculating(classifier, param_grid, X_train, X_test, y_train, y_test)

    else:
        pass


# Decision Tree function
def dt(X_train, y_train, X_test, y_test, pip_status):

    classifier = DecisionTreeClassifier()  # Change SVC() with your classifier
    if pip_status:
        param_grid = {
            'vectorizer__ngram_range': [(1, 1)], #(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3) TRY EACH ONE
            'transformer__use_idf': [False, True],
            'transformer__norm': ['l1', 'l2'],
            'classifier__max_depth': [2, 4, 6, 8, 10],
            'classifier__min_samples_split': [2, 4, 6, 8, 10],
            'classifier__min_samples_leaf': [1, 2, 3, 4, 5]
            # add yours parameters here (Note: you must write it like this: 'classifier__{name of the parameter}': [ ]
        }

        calculating(classifier, param_grid, X_train, X_test, y_train, y_test)

    else:
        pass


# Random Forest function
def rf(X_train, y_train, X_test, y_test, pip_status):


    classifier = SVC()  # Change SVC() with your classifier
    if pip_status:
        param_grid = {
            'vectorizer__ngram_range': [(1, 1)], #(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3) TRY EACH ONE
            'transformer__use_idf': [False, True],
            'transformer__norm': ['l1', 'l2'],
            # add yours parameters here (Note: you must write it like this: 'classifier__{name of the parameter}': [ ]
        }

        calculating(classifier, param_grid, X_train, X_test, y_train, y_test)

    else:
        pass


# preparing the dataset
def prepare():
    # Read CSV file(Dataset)
    df = pd.read_csv("finalDataset.csv", encoding="utf-8-sig") # read dataset

    # Encode comment and preparing X
    X = df["Comment"]

    # converting labels to numbers and preparing y
    le = preprocessing.LabelEncoder()
    le.fit(df.Label)
    df["Label"] = le.transform(df.Label)
    y = df["Label"]
    return X,y



def print_result(y_test, y_pred, best_score, best_param):
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print("The accuracy score of model: ",metrics.accuracy_score(y_test, y_pred))
    print("-----------------------------------------")
    # Print the best parameters and the best score
    print("Best score: ", best_score)
    print("Best parameters  for the given ngram : ")




# TF-IDF feature extraction function
def tfidf(classifier):
    X, y = prepare() # call prepare function
    tfidf_vectorizer = TfidfVectorizer(tokenizer=pyarabic.araby.tokenize, use_idf=True)
    featured_comment = tfidf_vectorizer.fit_transform(X)
    pip_status = False
    training(classifier, featured_comment, y, pip_status) # call training function

# N-Grams feature extraction function
def n_gram(classifier, n, m):
    X, y = prepare() # call prepare function
    vectorizer = CountVectorizer(ngram_range=(n,m))
    featured_comment = vectorizer.fit_transform(X)
    pip_status = False
    training(classifier, featured_comment, y, pip_status) # call training function


def pipeline(classifier):
    X, y = prepare()
    pip_status = True
    training(classifier,X,y, pip_status)


# This function to start the process of training
def start(classifier, feature_extraction, n, m):
    if feature_extraction == "ngram":
        n_gram(classifier, n, m)
    elif feature_extraction == "tfidf":
        tfidf(classifier)
    elif feature_extraction == "pipeline":
        pipeline(classifier)
    else:
        print("Wrong feature extraction input")


# First parameter choose one of these classifiers: SVM, NB, LR, KNN, DT, RF
# Second parameter choose one of these feature extraction: ngram, tfidf
# Third parameter enter n-gram level: 1, 2, 3 (this parameter will be ignored if TF-IDF choosen, but must enter any number)
start("DT","pipeline",1,2)