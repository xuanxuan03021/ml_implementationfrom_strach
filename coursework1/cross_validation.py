import os
import numpy as np
from numpy.random import default_rng
from classification import DecisionTreeClassifier
import evaluation

def k_fold_split(n_splits, n_instances, random_generator=default_rng()):
    # generate a random permutation of indices from 0 to n_instances
    shuffled_indices = random_generator.permutation(n_instances) 
    
    # split shuffled indices into almost equal sized splits
    split_indices = np.array_split(shuffled_indices, n_splits)
    
    return split_indices


def train_test_k_fold(n_folds, n_instances):
    # split the dataset into k splits
    split_indices = k_fold_split(n_folds, n_instances)
    folds = []
    for k in range(n_folds):
        test_indices = split_indices[k]
        train_indices = np.hstack(split_indices[:k] + split_indices[k+1:])
        folds.append([train_indices, test_indices])
    
    return folds

def show_accuracy(accuracies):
    print("Accuracies on each test fold: ")
    print(accuracies)

    print("Mean Accuracy: ", accuracies.mean())
    print("Std: ", accuracies.std())


def cross_validate(n_folds,x,y):
    
    assert x.shape[0] == y.shape[0]

    # accuracies on each fold
    accuracies = np.zeros((n_folds, ))

    #save each classifier 
    models = []

    for i, (train_indices, test_indices) in enumerate(train_test_k_fold(n_folds,len(x))):
        # Get the dataset from the correct splits
        x_train = x[train_indices, :]
        y_train = y[train_indices]
        x_test = x[test_indices, :]
        y_test = y[test_indices]

        classifier = DecisionTreeClassifier()
        if classifier.is_trained == False:

            classifier.fit(x_train, y_train) 
            predictions = classifier.predict(x_test) 
            accuracies[i] = evaluation.accuracy(y_test, predictions)
        
        models.append(classifier)
    
    show_accuracy(accuracies)

    return (models,accuracies)

def vote_fit(modelslist, x):
    length = len(modelslist)
    predictions = []

    for model in modelslist:
        if not model.is_trained:
            raise Exception("DecisionTreeClassifier has not yet been trained.")
        predictions.append(model.predict(x))
        
    predictions  = np.array(predictions)
        
    final_predictions = np.zeros((x.shape[0],),dtype=np.object)
    for i in range(x.shape[0]):
        results = np.zeros(length,dtype=np.object)
        for j in range(length):
            results[j] = predictions[j][i]
        classes, counts = np.unique(results, return_counts=True)
        d = dict(zip(classes,counts))
        final_predictions[i] = max(d, key=lambda k: d[k])
        
    return final_predictions












