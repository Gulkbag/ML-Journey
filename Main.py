import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

'''best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    
    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > best:
        best = acc
        with open("student-model.pickle", "wb") as f:
            pickle.dump(linear, f)'''

model_pickle = open("student-model.pickle", "rb")
linear = pickle.load(model_pickle)

print("Co: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)
print("=================================")
print("Predictions: \n")

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print("Predicted Grade : " ,predictions[x], "Student Attribs : ",x_test[x], "Actual Grade : ", y_test[x])

p = 'failures'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()