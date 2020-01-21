import pandas
from utils.file_helper import FileHelper
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


class Classification:
    def __init__(self, train_file, test_file=None):
        self.X_train = []
        self.y_train = []
        self.y_test = []
        self.X_test = []

        self.read_train(train_file)
        if test_file is not None:
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
        print("\tMultinomialNB Precision: ", metrics.precision_score(self.y_test, predicted))
        print("\tMultinomialNB Recall: ", metrics.recall_score(self.y_test, predicted))

    def gaussianNaiveBayes(self):
        print("Running Gaussian Naive Bayes")

        nb = GaussianNB().fit(self.X_train, self.y_train)
        predicted = nb.predict(self.X_test)

        print("\tGaussianNB Accuracy: ", metrics.accuracy_score(self.y_test, predicted))
        print("\tGaussianNB Precision: ", metrics.precision_score(self.y_test, predicted))
        print("\tGaussianNB Recall: ", metrics.recall_score(self.y_test, predicted))

    def bernoulliNaiveBayes(self):
        print("Running Bernoulli Naive Bayes")

        nb = BernoulliNB().fit(self.X_train, self.y_train)
        predicted = nb.predict(self.X_test)

        print("\tBernoulliNB Accuracy: ", metrics.accuracy_score(self.y_test, predicted))
        print("\tBernoulliNB Precision: ", metrics.precision_score(self.y_test, predicted))
        print("\tBernoulliNB Recall: ", metrics.recall_score(self.y_test, predicted))

    def knn(self, bagging=False):
        print("Running KKN")

        knn = None
        if bagging:
            knn = BaggingClassifier(KNeighborsClassifier(n_neighbors=5), n_jobs=-1)
            knn.fit(self.X_train, self.y_train)
            
            predicted = knn.predict(self.X_test)

            print("\tKNN Bagging Accuracy: ", metrics.accuracy_score(self.y_test, predicted))
            print("\tKNN Bagging Precision: ", metrics.precision_score(self.y_test, predicted))
            print("\tKNN Bagging Recall: ", metrics.recall_score(self.y_test, predicted))

        else:
            knn = KNeighborsClassifier(n_neighbors=5, algorithm='auto', n_jobs=-1)
            knn.fit(self.X_train, self.y_train)
            
            predicted = knn.predict(self.X_test)

            print("\tKNN Accuracy: ", metrics.accuracy_score(self.y_test, predicted))
            print("\tKNN Precision: ", metrics.precision_score(self.y_test, predicted))
            print("\tKNN Recall: ", metrics.recall_score(self.y_test, predicted))

    def decision_tree(self):
        print("Running Decision Tree")
        decision_tree = DecisionTreeClassifier()

        decision_tree.fit(self.X_train, self.y_train)

        predicted = decision_tree.predict(self.X_test)

        print("\tDecision Tree Accuracy: ", metrics.accuracy_score(self.y_test, predicted))
        print("\tDecision Tree Precision: ", metrics.precision_score(self.y_test, predicted))
        print("\tDecision Tree Recall: ", metrics.recall_score(self.y_test, predicted))

    def svm(self):
        print("Running SVM")
        #svm_parameters = {'kernel': ('linear', 'rbf'), 'C':[1,5]}
        svc = SVC(probability=True, kernel='rbf')
        #svm_classifier = GridSearchCV(svc, svm_parameters)
        #svm_classifier.fit(self.X_train, self.y_train)
        svc.fit(self.X_train, self.y_train)
        
        predicted = svc.predict(self.X_test)

        print("\tSVM - RBF Accuracy: ", metrics.accuracy_score(self.y_test, predicted))
        print("\tSVM - RBF Precision: ", metrics.precision_score(self.y_test, predicted))
        print("\tSVM - RBF Recall: ", metrics.recall_score(self.y_test, predicted))

        svc = SVC(probability=True, kernel='linear')
        #svm_classifier = GridSearchCV(svc, svm_parameters)
        #svm_classifier.fit(self.X_train, self.y_train)
        svc.fit(self.X_train, self.y_train)
        
        predicted = svc.predict(self.X_test)

        print("\tSVM - Linear Accuracy: ", metrics.accuracy_score(self.y_test, predicted))
        print("\tSVM - Linear Precision: ", metrics.precision_score(self.y_test, predicted))
        print("\tSVM - Linear Recall: ", metrics.recall_score(self.y_test, predicted))

    def multilayer_perceptron(self):
        print("Running Voting MLP")
        mlp = BaggingClassifier(MLPClassifier(solver='lbfgs', alpha=1e-5, activation=('logistic'), hidden_layer_sizes=(30), learning_rate_init=0.001, max_iter=100000, random_state=1))
        
        mlp.fit(self.X_train, self.y_train)

        predicted = mlp.predict(self.X_test)
        print("\tMLP: ", metrics.accuracy_score(self.y_test, predicted))


    def voting_classifier(self):
        print("Running Voting Classifier")
        classiers = [
            ('multinomial', MultinomialNB()),
            ('bernoulli', BernoulliNB()),
            ('svm', SVC(probability=True, kernel='linear'))
        ]

        voting_classifier = VotingClassifier(classiers, n_jobs=-1, weights=[1, 2, 2])

        voting_classifier.fit(self.X_train, self.y_train)

        predicted = voting_classifier.predict(self.X_test)
        print("\tVoting Accuracy: ", metrics.accuracy_score(self.y_test, predicted))
        print("\tVoting Precision: ", metrics.precision_score(self.y_test, predicted))
        print("\tVoting Recall: ", metrics.recall_score(self.y_test, predicted))

    def predict_comment(self, features):
        nb = BernoulliNB().fit(self.X_train, self.y_train)
        predicted = nb.predict(features)

        print("Classification: {}".format(predicted[0]))