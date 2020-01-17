import pandas
from utils.file_helper import FileHelper
from sklearn.naive_bayes import MultinomialNB
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
        clf = MultinomialNB().fit(self.X_train, self.y_train)
        predicted = clf.predict(self.X_test)
        print("MultinomialNB Accuracy: ", metrics.accuracy_score(self.y_test, predicted))