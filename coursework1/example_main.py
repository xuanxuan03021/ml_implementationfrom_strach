##############################################################################
# 60012: Introduction to Machine Learning
# Coursework 1 example execution code
# Prepared by: Josiah Wang
##############################################################################

import numpy as np
import pydot
from classification import DecisionTreeClassifier
from improvement import train_and_predict
from evaluation import evaluate
from cross_validation import cross_validate, vote_fit
from plot import plot_tree

if __name__ == "__main__":
    print("Loading the training dataset...");
    x = np.array([
            [5,7,1],
            [4,6,2],
            [4,6,3], 
            [1,3,1], 
            [2,1,2], 
            [5,2,6]
        ])
    
    y = np.array(["A", "A", "A", "C", "C", "C"])
    
    print("Training the decision tree...")
    classifier = DecisionTreeClassifier()
    classifier.fit(x, y)

    print("Loading the test set...")

    x_test = np.array([
            [1,6,3],
            [0,5,5],
            [1,5,0],
            [2,4,2]
        ])

    y_test = np.array(["A", "A", "C", "C"])

    print("Making predictions on the test set...")
    predictions = classifier.predict(x_test)
    print("Predictions: {}".format(predictions))
    print("The classify rule is",classifier.node)
    classes = ["A", "C"];

    print("Pruning the decision tree...")
    x_val = np.array([
                [6,7,2],
                [3,1,3]
            ])
    y_val = np.array(["A", "C"])

    classifier.prune(x_val, y_val)
    print("The classify rule is",classifier.node)

    print("Making predictions on the test set using the pruned decision tree...")
    predictions = classifier.predict(x_test)
    print("Predictions: {}".format(predictions))

    # print("Making predictions on the test set using the improved decision tree...")
    # predictions = train_and_predict(x, y, x_test, x_val, y_val)
    # print("Predictions: {}".format(predictions))

    print("============Train in train_full============")
    x_full = np.loadtxt("./data/train_full.txt", delimiter=',', dtype=str)
    y_full = x_full[:, 16]
    x_full = x_full[:, 0:16].astype(int)
    print("Training the decision tree train_full...")
    classifier = DecisionTreeClassifier()
    classifier.fit(x_full, y_full)
    print(classifier.node)
    predictions = classifier.predict(x_full)
    evaluate(y_full,predictions)
    print("Plot the tree in train_full")
    plot_tree(classifier.node)
    print("============test in train_full============")
    x_test_full = np.loadtxt("./data/test.txt", delimiter=',', dtype=str)
    x_test=x_test_full[:,0:16].astype(int)
    y_test=x_test_full[:,16]
    print("Making predictions on the test set （train_full）...")
    predictions = classifier.predict(x_test)
    print("Predictions: {}".format(predictions))
    print("============evaluation in train_full============")
    evaluate(y_test,predictions)

    print("============Cross Validation============")
    n_folds = 10
    #save each classifier
    [models,accuracies] = cross_validate(n_folds,x_full,y_full)

    print("try the decision tree with the highest accuracy from 10 fold cross-validation on testset")
    results = dict(zip(models,accuracies))
    bestmodel = max(results, key=lambda k: results[k])
    predictions = bestmodel.predict(x_test)
    print("Performance of the best model: ")
    print(evaluate(y_test, predictions))

    print("combining the predictions on test.txt for all 10 decision trees")
    voted_predictions = vote_fit(models,x_test)
    print("Evaluating voted predictions: ")
    evaluate(y_test,voted_predictions)

    print("============pruning  in train_full============")
    x_val_full = np.loadtxt("./data/validation.txt", delimiter=',', dtype=str)
    x_val=x_val_full[:,0:16].astype(int)
    y_val=x_val_full[:,16]
    classifier.prune(x_val,y_val)
    print("After pruning The classify rule is",classifier.node)
    print("Making predictions on the validation set using the pruned decision tree...")
    predictions = classifier.predict(x_val)
    print("Predictions: {}".format(predictions))
    evaluate(y_val,predictions)

    print("Making predictions on the test set using the pruned decision tree...")
    predictions = classifier.predict(x_test)
    print("Predictions: {}".format(predictions))
    evaluate(y_test,predictions)


    #============================================================train_sub===========================================
    print("============Train in train_sub============")
    x_sub = np.loadtxt("./data/train_sub.txt", delimiter=',', dtype=str)
    y_sub = x_sub[:, 16]
    x_sub = x_sub[:, 0:16].astype(int)
    print("Training the decision tree train_sub...")
    classifier = DecisionTreeClassifier()
    classifier.fit(x_sub, y_sub)
    print("============test in train_sub============")
    x_test_sub_full = np.loadtxt("./data/test.txt", delimiter=',', dtype=str)
    x_test_sub=x_test_sub_full[:,0:16].astype(int)
    y_test_sub=x_test_sub_full[:,16]

    print("Making predictions on the test set（train_sub）...")
    predictions = classifier.predict(x_test_sub)
    print("Predictions: {}".format(predictions))

    print("============evaluation in train_sub============")

    evaluate(y_test_sub,predictions)

#============================================================train_noisy===========================================

    print("============Train in train_noisy============")
    x_noisy = np.loadtxt("./data/train_noisy.txt", delimiter=',', dtype=str)
    y_noisy = x_noisy[:, 16]
    x_noisy = x_noisy[:, 0:16].astype(int)
    print("Training the decision tree train_noisy...")
    classifier = DecisionTreeClassifier()
    classifier.fit(x_noisy, y_noisy)
    print("The classify rule is", classifier.node)
    predictions = classifier.predict(x_noisy)
    evaluate(y_noisy,predictions)

    print("============test in train_noisy============")
    x_test_noisy = np.loadtxt("./data/test.txt", delimiter=',', dtype=str)
    y_test_noisy=x_test_noisy[:,16]
    x_test_noisy=x_test_noisy[:,0:16].astype(int)
    print("Making predictions on the test set （train_noisy）...")
    predictions = classifier.predict(x_test_noisy)
    print("Predictions: {}".format(predictions))

    print("============evaluation in train_noisy============")
    evaluate(y_test_noisy,predictions)

    print("============pruning in train_noisy============")
    classifier.prune(x_val, y_val)
    print("After pruning the classify rule is", classifier.node)
    print("Making predictions on the validation set using the pruned decision tree...")
    predictions = classifier.predict(x_val)
    print("Predictions: {}".format(predictions))
    evaluate(y_val, predictions)

    print("Making predictions on the test set using the pruned decision tree...")
    predictions = classifier.predict(x_test_noisy)
    print("Predictions: {}".format(predictions))
    evaluate(y_test_noisy, predictions)





