import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

#print(cancer.feature_names)
#print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

classes = ['malignant' 'benign']

svmc = svm.SVC(kernel="linear", C=2)
svmc.fit(x_train, y_train)

y_pred = svmc.predict(x_test)

acc_svm = metrics.accuracy_score(y_test, y_pred)

print("SVM Accuracy: ",acc_svm)
