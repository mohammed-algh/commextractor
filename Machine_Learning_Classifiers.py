import pandas as pd
from Preprocessing import doPreprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC

# Classifiers:   SVM ,  NB ,  LR , KNN , DT , RF
# Feature extraction: N-Gram , TF-IDF



dataset = pd.read_csv("finalDataset.csv", encoding="utf-8-sig") # read dataset
df = pd.DataFrame(dataset, columns=["Comment", "Label"]) #create dataframe to manipulate the dataset

# do preprocessing
for i in range(len(df)):
    df["Comment"][i] = doPreprocessing(df["Comment"][i])

# convert labels into numbers
# 0 for negative
# 1 for netural
# 2 for positive
for i in range(len(df)):
    df["Label"][i] = df["Label"][i]
    if df["Label"][i] == "Negitive":
        df["Label"][i] = 0
    elif df["Label"][i] == "Netural":
        df["Label"][i] = 1
    elif df["Label"][i] == "Positive":
        df["Label"][i] = 2




X = df.drop("Label",axis=1) # For Comment column (drop Label column)
y = df["Label"] # For Label column



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=42) # Split data into 70% training and 30% testing


svmClassf = SVC(kernel="linear") # create the classifier SVM
svmClassf.fit(X_train,y_train) # train the classifier
y_pred = svmClassf.predict(X_test) # test phase
print(y_pred) # print the prediction
