import pandas
from utils.file_helper import FileHelper
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics

class Classification:
    def __init__(self, train_file, test_file):
        self.X_train = []
        self.y_train = []
        self.y_test = []
        self.X_test = []

        self.read_train(train_file)
        self.read_test(test_file)

    def read_train(self, train_file):
        data_read = FileHelper.read_from_file(train_file)
        for row in data_read.split("\n"):
            if len(row) == 0:
                continue
            row_split = row.split(" ")
            self.y_train.append(int(row_split[-1]))
            train_row = []
            row_split.pop()
            for row_value in row_split:
                train_row.append(float(row_value))
            self.X_train.append(train_row)

    def read_test(self, test_file):
        data_read = FileHelper.read_from_file(test_file)
        for row in data_read.split("\n"):
            if len(row) == 0:
                continue
            row_split = row.split(" ")
            self.y_test.append(int(row_split[-1]))
            row_split.pop()
            test_row = []
            for row_value in row_split:
                test_row.append(float(row_value))
            self.X_test.append(test_row)

    def multinomialNaiveBayes(self):
        print("Running Multinomial Naive Bayes")

        nb = MultinomialNB().fit(self.X_train, self.y_train)
        predicted = nb.predict(self.X_test)

        print("\tMultinomialNB Accuracy: ", metrics.accuracy_score(self.y_test, predicted))

    def gaussianNaiveBayes(self):
        print("Running Gaussian Naive Bayes")

        nb = GaussianNB().fit(self.X_train, self.y_train)
        predicted = nb.predict(self.X_test)

        print("\tMultinomialNB Accuracy: ", metrics.accuracy_score(self.y_test, predicted))

    def bernoulliNaiveBayes(self):
        print("Running Bernoulli Naive Bayes")

        nb = BernoulliNB().fit(self.X_train, self.y_train)
        predicted = nb.predict(self.X_test)

        print("\tMultinomialNB Accuracy: ", metrics.accuracy_score(self.y_test, predicted))

    def knn(self, bagging=False):
        print("Running KKN")

        knn = None
        #if bagging:
        knn = BaggingClassifier(KNeighborsClassifier(n_neighbors=5), n_jobs=-1)
        knn.fit(self.X_train, self.y_train)
        
        predicted = knn.predict(self.X_test)

        print("\tKNN Bagging: ", metrics.accuracy_score(self.y_test, predicted))

        #else:
        knn = KNeighborsClassifier(n_neighbors=5, algorithm='auto', n_jobs=-1)
        knn.fit(self.X_train, self.y_train)
        
        predicted = knn.predict(self.X_test)

        print("\tKNN: ", metrics.accuracy_score(self.y_test, predicted))

    def svm(self):
        print("Running SVM")
        #svm_parameters = {'kernel': ('linear', 'rbf'), 'C':[1,5]}
        svc = SVC(probability=True, kernel='rbf')
        #svm_classifier = GridSearchCV(svc, svm_parameters)
        #svm_classifier.fit(self.X_train, self.y_train)
        svc.fit(self.X_train, self.y_train)
        
        predited = svc.predict(self.X_test)

        print("\tSVM - RBF: ", metrics.accuracy_score(self.y_test, predited))

        svc = SVC(probability=True, kernel='linear')
        #svm_classifier = GridSearchCV(svc, svm_parameters)
        #svm_classifier.fit(self.X_train, self.y_train)
        svc.fit(self.X_train, self.y_train)
        
        predited = svc.predict(self.X_test)

        print("\tSVM - RBF: ", metrics.accuracy_score(self.y_test, predited))