from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from model_abstract import Model_Abstract
from sklearn.model_selection import KFold
class Sklearn_model(Model_Abstract):
    def __init__(self, helper, model, k_fold_splits=5, name = 'Sklearn Model'):
        super().__init__(helper, model, name)
        self.clf = self.model
        self.k_fold_splits = k_fold_splits

    def fit(self, X, Y):
        self.clf.fit(X ,Y)

    def predict(self, X):
        return self.clf.predict(X)

    def print_classification_report(self, Y, Y_pred):
        print("Classification report for - \n{}:\n{}\n".format(self.clf, metrics.classification_report(Y, Y_pred)))

    def evaluate_model(self, X, Y):
        balanced_k_fold = self.helper.balanced_k_fold_splits(Y, self.k_fold_splits)
        skf = StratifiedKFold(n_splits=self.k_fold_splits, shuffle=False)
        skf.get_n_splits(X, Y)
        kf = KFold(n_splits=self.k_fold_splits, shuffle=False)
        kf.get_n_splits(X, Y)
        k_val_accuracy = []
        k_train_accuracy = []

        print("############### Running validation with Balanced K folds  ###############")
        cross_train_total, cross_val_total = self.evaluate_folds(X, Y, balanced_k_fold)
        k_val_accuracy.append(cross_val_total / (self.k_fold_splits))
        k_train_accuracy.append(cross_train_total / (self.k_fold_splits))
        print("With Balanced folds: K fold validation accuracy = " + str(cross_val_total / (self.k_fold_splits)) + "%")
        print("With Balanced folds: K fold training accuracy = " + str(cross_train_total / (self.k_fold_splits)) + "%")

        print("############### Running validation with Stratified K folds  ###############")
        cross_train_total, cross_val_total = self.evaluate_folds(X, Y, skf.split(X, Y))
        k_val_accuracy.append(cross_val_total / (self.k_fold_splits))
        k_train_accuracy.append(cross_train_total / (self.k_fold_splits))
        print("With Stratified folds: K fold validation accuracy = " + str(cross_val_total / (self.k_fold_splits)) + "%")
        print("With Stratified folds: K fold training accuracy = " + str(cross_train_total / (self.k_fold_splits)) + "%")

        print("############### Running validation with K folds  ###############")
        cross_train_total, cross_val_total = self.evaluate_folds(X, Y, kf.split(X, Y))
        k_val_accuracy.append(cross_val_total / (self.k_fold_splits))
        k_train_accuracy.append(cross_train_total / (self.k_fold_splits))
        print("With K folds: K fold validation accuracy = " + str(cross_val_total / (self.k_fold_splits)) + "%")
        print("With K folds: K fold training accuracy = " + str(cross_train_total / (self.k_fold_splits)) + "%")

        self.k_val_accuracy = min(k_val_accuracy)
        self.k_train_accuracy = min(k_train_accuracy)
        print("***********************************************************************")
        print("Worst: K fold validation accuracy = " + str(self.k_val_accuracy) + "%")
        print("Worst: K fold training accuracy = " + str(self.k_val_accuracy) + "%")



    def evaluate_folds(self, X, Y, folds):
        cross_val_total = 0
        cross_train_total = 0
        k = 0
        for train_index, test_index in folds:

            X_train, X_val, Y_train, Y_val = X[train_index], X[test_index], Y[train_index], Y[test_index]
            self.clf.fit(X_train, Y_train)
            pred = self.clf.predict(X_val)
            pred_train = self.clf.predict(X_train)
            val_accuracy = self.helper.calc_accuracy(Y_val, pred)
            train_accuracy = self.helper.calc_accuracy(Y_train, pred_train)
            cross_val_total += float(val_accuracy)
            cross_train_total += float(train_accuracy)
            print("In fold = " + str(k) + ", val accuracy = " + str(val_accuracy))
            print("In fold = " + str(k) + ", train accuracy = " + str(train_accuracy))
            self.print_classification_report(Y_val, pred)
            k += 1
        return cross_train_total, cross_val_total


