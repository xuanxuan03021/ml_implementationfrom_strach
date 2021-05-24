#############################################################################
# 60012: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the fit(), predict() and prune() methods of
# DecisionTreeClassifier. You are free to add any other methods as needed. 
##############################################################################

import numpy as np


class DecisionTreeClassifier(object):
    """ Basic decision tree classifier
    
    Attributes:
    is_trained (bool): Keeps track of whether the classifier has been trained
    
    Methods:
    fit(x, y): Constructs a decision tree from data X and label y
    predict(x): Predicts the class label of samples X
    prune(x_val, y_val): Post-prunes the decision tree
    """

    def __init__(self):
        self.is_trained = False
        self.node={}#trained model
        self.x_train=np.array([])# features of the train dataset
        self.y_train = np.array([])# labels of the train dataset

    # a helper function to calculate the entropy of a dataset
    def caluculate_dataset_entropy(self,y):

        # y (numpy.ndarray): Class labels, numpy array of shape (N, )
        #                    Each element in y is a str
        num_instance = y.shape[0]

        #the unique classes in a dataset
        classes = set(y)
        every_label_counts = {}
        for label in classes:
            every_label_counts[label] = len(np.where(y == label)[0])
        entropy = 0
        for key in every_label_counts:

            #the probability of each class in the dataset
            prob = every_label_counts[key] * 1.0 / num_instance

            #calculate the entropy
            entropy -= prob * np.log2(prob)
        return entropy


    def fit(self, x, y):
        """ Constructs a decision tree classifier from data
        
        Args:
        x (numpy.ndarray): Instances, numpy array of shape (N, K) 
                           N is the number of instances
                           K is the number of attributes
        y (numpy.ndarray): Class labels, numpy array of shape (N, )
                           Each element in y is a str 
        """
        
        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."
        # remember train instances
        self.x_train=x
        self.y_train=y
        #######################################################################
        #                 ** TASK 2.1: COMPLETE THIS METHOD **
        #######################################################################    
        def train_model(x, y):

            # ==================base case================
            # 1. if the database only have one class, then return the class label
            # 2. all the features of the instances are the same, then retuen the majority of the class

            same_attribute = 0
            if len(set(y)) == 1:
                return y[0]
            for i in range(x.shape[1]):
                if len(set(x[:, i])) == 1:
                    same_attribute += 1
            if same_attribute == x.shape[1]:

                #get the unqiue classes and the position of them
                classes, pos = np.unique(y, return_inverse=True)
                # count the occurancy of each class
                counts = np.bincount(pos)
                # get the majority label index
                label=counts.argmax()
                return classes[label]

            # ==================recursive part===============
            best_feature_index = -1
            best_feature_split_value = 0
            max_info_gain = 0

            #calculate the entropy of the entire database
            entropy = self.caluculate_dataset_entropy(y)

            # find the best split feature
            for i in range(x.shape[1]):

                # sort the data according to the feature
                num_instance = y.shape[0]
                x_rank = x[:, i].argsort()
                x_sorted = x[x_rank]
                y_sorted = y[x_rank]
                split_value = -1
                for j in range(y.shape[0] - 1):

                    #if the lable change between two adjanct instance and the feature value is not the last split value
                    if (y_sorted[j] != y_sorted[j + 1] and split_value != x_sorted[j][i]):
                        split_value = x_sorted[j][i]

                        #calculate the entropy of the smaller part (<= split value)
                        prob_front = len(np.where(x[:, i] <= split_value)[0]) * 1.0 / num_instance
                        y_front = y[np.where(x[:, i] <= split_value)]
                        prob_front_entropy = self.caluculate_dataset_entropy(y_front)

                        #calculate the entropy of the bigger part (> split value)
                        prob_end = len(np.where(x[:, i] > split_value)[0]) * 1.0 / num_instance
                        y_end = y[np.where(x[:, i] > split_value)]
                        prob_end_entropy = self.caluculate_dataset_entropy(y_end)

                        #calculate the information gain of the feature
                        feature_entropy = prob_front * prob_front_entropy + prob_end * prob_end_entropy
                        info_gain = entropy - feature_entropy

                        #select the feature which have the highest information gain
                        if (info_gain > max_info_gain):
                            max_info_gain = info_gain
                            best_feature_split_value = split_value
                            best_feature_index = i

            # add it in the tree
            node = {best_feature_index: {}}

            # recursively call the function (less than split value)
            node[best_feature_index]["<" + str(best_feature_split_value)] = train_model(
                x[x[:, best_feature_index] <= best_feature_split_value],
                y[x[:, best_feature_index] <= best_feature_split_value])

            # recursively call the function (bigger than split value)
            node[best_feature_index][">" + str(best_feature_split_value)] = train_model(
                x[np.where(x[:, best_feature_index] > best_feature_split_value)],
                y[np.where(x[:, best_feature_index] > best_feature_split_value)])

            # set a flag so that we know that the classifier has been trained
            self.is_trained = True
            return node
        # keep the trained model in the classifier
        self.node=train_model(x,y)

    # a helper function to find the class label
    def find_class(self,one_row,node):
        #=========base case=============
        #if it is a class label then return
        if type(node).__name__ != 'dict':
            return node
        #=========recursive part==========
        # split attribute
        attribute = list(node.keys())[0]
        # split value of the split attribute
        split_point = int(list(node[attribute].keys())[0][1:])

        #left_sub tree, recursively find the class label
        if (one_row[attribute] <= split_point):
            split_key = list(node[attribute].keys())[0]
            return self.find_class(one_row, node[attribute][split_key])

        #right_sub tree, recursively find the class label
        else:
            split_key = list(node[attribute].keys())[1]
            return self.find_class(one_row, node[attribute][split_key])


    def predict(self, x):
        """ Predicts a set of samples using the trained DecisionTreeClassifier.
        
        Assumes that the DecisionTreeClassifier has already been trained.
        
        Args:
        x (numpy.ndarray): Instances, numpy array of shape (M, K) 
                           M is the number of test instances
                           K is the number of attributes
        
        Returns:
        numpy.ndarray: A numpy array of shape (M, ) containing the predicted
                       class label for each instance in x
        """
        
        # make sure that the classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("DecisionTreeClassifier has not yet been trained.")
        
        # set up an empty (M, ) numpy array to store the predicted labels 
        # feel free to change this if needed
        predictions = np.zeros((x.shape[0],), dtype=np.object)

        #######################################################################
        #                 ** TASK 2.2: COMPLETE THIS METHOD **
        #######################################################################
        for i in range(x.shape[0]):
            # predict the class of every instance
            predictions[i]=self.find_class(x[i,:],self.node)
        return predictions
        

    def prune(self, x_val, y_val):
        """ Post-prune your DecisionTreeClassifier given some optional validation dataset.

        You can ignore x_val and y_val if you do not need a validation dataset for pruning.

        Args:
        x_val (numpy.ndarray): Instances of validation dataset, numpy array of shape (L, K).
                           L is the number of validation instances
                           K is the number of attributes
        y_val (numpy.ndarray): Class labels for validation dataset, numpy array of shape (L, )
                           Each element in y is a str 
        """
        
        # make sure that the classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("DecisionTreeClassifier has not yet been trained.")



        #######################################################################
        #                 ** TASK 4.1: COMPLETE THIS METHOD **
        #######################################################################

        # a helper function: check if we want to prune the node
        # if the errors of the prune tree is smaller or equal to the original tree then prune (return the majority of the class of that part of the dataset)
        #else return the original node
        def pruning(x_val,y_val,y_train,node):
            predictions=np.zeros((x_val.shape[0],), dtype=np.object)
            for i in range(x_val.shape[0]):
                predictions[i] = self.find_class(x_val[i, :], node)

            #error of the original node
            error_node = sum(predictions != y_val)

            # Finds all unique elements and their positions
            classes, pos = np.unique(y_train, return_inverse=True)
            # Count the number of each unique element
            counts = np.bincount(pos)
            # Finds the positions of the maximum count
            maxpos = counts.argmax()
            leaf = classes[maxpos]

            #error of the leaf
            error_leaf = sum(leaf != y_val)

            #compare the error
            if error_leaf <= error_node:
                return leaf
            else:
                return node

        #=============base case=============
        def post_pruning(x_val,y_val,x_train,y_train,node):
            #splite attribute
            key=list(node.keys())[0]
            left_value=list(node[key].keys())[0]
            right_value=list(node[key].keys())[1]
            sub_node=node[key]

            # if all children nodes are leaves
            if not isinstance(sub_node[left_value],dict) and not isinstance(sub_node[right_value],dict):
                return pruning(x_val,y_val,y_train,node)

        #=============recursive part==========

            # split the bigger part of the train and validation dataset (split attribute> split value)
            split_value=int(left_value[1:])
            sub_x_train_bigger=x_train[x_train[:,key]>split_value]
            sub_y_train_bigger=y_train[x_train[:,key]>split_value]
            sub_x_val_bigger=x_val[x_val[:,key]>split_value]
            sub_y_val_bigger = y_val[x_val[:, key] > split_value]

            # split the smaller part of the train and validation dataset (split attribute<= split value)
            sub_x_train_smaller=x_train[x_train[:,key]<=split_value]
            sub_y_train_smaller = y_train[x_train[:, key] <= split_value]
            sub_x_val_smaller = x_val[x_val[:, key] <= split_value]
            sub_y_val_smaller = y_val[x_val[:, key] <= split_value]

            #if the child node is dict the recursively find the node which is not a dictionary(smaller part)
            if(isinstance(sub_node[left_value],dict)):
                node[key][left_value]=post_pruning(sub_x_val_smaller,sub_y_val_smaller,sub_x_train_smaller,sub_y_train_smaller,sub_node[left_value])

            #if the child node is dict the recursively find the node which is not a dictionary (bigger part)
            if(isinstance(sub_node[right_value], dict)):
                node[key][right_value]=post_pruning(sub_x_val_bigger,sub_y_val_bigger,sub_x_train_bigger,sub_y_train_bigger,sub_node[right_value])

            # if the node after pruning have two leaves then check if it can be pruned
            if not isinstance(sub_node[left_value],dict) and not isinstance(sub_node[right_value],dict):
                return pruning(x_val, y_val, y_train, node)
            return node


        post_pruning(x_val,y_val,self.x_train,self.y_train,self.node)


       


