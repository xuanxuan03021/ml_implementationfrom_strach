import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import r2_score
import torch.optim as optim
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, train_test_split


def activation_function(activation="identity"):
    if activation == "relu":
        return nn.ReLU()
    if activation == "sigmoid":
        return nn.Sigmoid()
    # assume identity function
    return nn.Identity()


def loss_function(loss_fun):
    if loss_fun == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    else:
        # "mse" or others default to MSE
        criterion = nn.MSELoss()
    return criterion


class NeuralNetwork(nn.Module):

    def __init__(self, n_input_vars, neurons=[], activations=[], n_output_vars=1):
        super(NeuralNetwork, self).__init__()
        self.neurons=neurons
        self.input_dim=n_input_vars
        self.output_var=n_output_vars
        self.num_layers=len(neurons)
        self.activations=activations
        self._layers=nn.ModuleList()

        for i in range(len(neurons)):
            # input layer
            if i == 0:
                self._layers.append(nn.Linear(n_input_vars, neurons[i]))
            # hidden layer
            else:
                self._layers.append(nn.Linear(neurons[i - 1], neurons[i]))

    def forward(self, x):
        for i in range(self.num_layers):
            x=self._layers[i](x)
            act_function=activation_function(self.activations[i])
            x=act_function(x)
        return x



class Regressor(BaseEstimator):

    def __init__(self, x, nb_epoch=10, batch_size=30, learning_rate=0.01, loss_fun="mse", neurons=[32,32, 1],
                 activations=["relu", "relu","identity"]):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.

        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape
                (batch_size, input_size), used to compute the size
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own

        self.x = x
        X, _ = self._preprocessor(x, training=True)

        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_fun = loss_fun
        self.neurons = neurons
        self.activations = activations
        # check the number of neurons and the number of activations
        if (len(neurons)!= len(activations)):
            print("The number of layers are not same as the number of activations!")
            exit(1)
        self.model = NeuralNetwork(n_input_vars=self.input_size, neurons=self.neurons, activations=self.activations, \
                                   n_output_vars=self.output_size)
        #######################################################################
        #                     ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y=None, training=False):
        """
        Preprocess input of the network.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size).
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).

        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Return preprocessed x and y, return None for y if it was None

        # missing value
        x = x.fillna(x.mean())
        # get textual list
        non_textual = x.select_dtypes(exclude=["object"])
        textual = x.select_dtypes(include=["object"])
        numerical_name = list(non_textual.columns)
        text_name = list(textual.columns)

        #if it is training set nomalize it
        if (training == True):
            ss = preprocessing.StandardScaler()
            scale_features = numerical_name
            x[scale_features] = ss.fit_transform(x[scale_features])
            self.mean = ss.mean_
            self.std = ss.var_

            # get dummy to deal with textual categorical data
            if text_name:
                x = pd.concat([x, pd.get_dummies(x[text_name], dummy_na=True)], axis=1)
                x.drop(text_name, axis=1, inplace=True)
            self.col_name = list(x.columns)

        #if it is not training set, just use the training set value to normalize
        else:
            x[numerical_name]=(x[numerical_name]-self.mean)/self.std
            # get dummy to deal with textual categorical data
            if text_name:
                x = pd.concat([x, pd.get_dummies(x[text_name], dummy_na=True)], axis=1, ignore_index=False)
                x.drop(text_name, axis=1, inplace=True)


            # if there are some categories lost
            if len(self.col_name) != len(list(x.columns)):
                for i in range(len(self.col_name)):
                    if (self.col_name[i] != list(x.columns)[i]):
                        print("the col",list(x.columns), type(list(x.columns)[0]))
                        print("The trainig set is ",self.col_name, type(list(x.columns)[0]))
                        x.insert(i, self.col_name[i], 0)
        # =====================y=====================
        if not y is None:
            y=y.apply(np.log)

        return x.values, (y.values if isinstance(y, pd.DataFrame) else None)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
    def shuffle(self,input_dataset, target_dataset):
        """
        Returns shuffled versions of the inputs.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_data_points, n_features) or (#_data_points,).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_data_points, #output_neurons).

        Returns:
            - {np.ndarray} -- shuffled inputs.
            - {np.ndarray} -- shuffled_targets.
        """
        seed=3
        rg=np.random.default_rng(seed)
        shuffled_indices=rg.permutation(input_dataset.shape[0])
        return input_dataset[shuffled_indices],target_dataset[shuffled_indices]


    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Apply preprocessing before converting to tensor

        print("begin predict")
        temp_x, temp_y = self._preprocessor(x, y=y)
        print(temp_x)
        print(temp_y)
        num_input=temp_x.shape[0]

        x_train_tensor = torch.from_numpy(temp_x).float()
        y_train_tensor = torch.from_numpy(temp_y).float()

        # should be "None" at the moment. It will only be filled later after you call backward()

        print("The architecture of the neural network is: ",self.model)
        print("The epoch: ",self.nb_epoch)
        criterion = loss_function(self.loss_fun)
        print("The criterion",criterion)
        print("learning_rate",self.learning_rate)
        optimiser = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        num_batch = int(np.floor( num_input/ self.batch_size))

        #mini batch  shuffle
        for epoch in range(self.nb_epoch):
            #shuffle the dataset
            input_dataset, target_dataset = self.shuffle(x_train_tensor, y_train_tensor)
            #for every batch
            for j in range(num_batch):
                    # forward pass
                    y_hat = self.model(input_dataset[j * self.batch_size:(j + 1) * self.batch_size, :])
                    batch_y_ground = target_dataset[j * self.batch_size:(j + 1) * self.batch_size, :]
                    # compute loss
                    loss = criterion(y_hat, batch_y_ground)

                    # Reset the gradients to zero
                    optimiser.zero_grad()

                    # Backward pass (compute the gradients)
                    loss.backward()

                    # update parameters
                    optimiser.step()

                    #print(f"Epoch: {epoch}\t batch {j} \t Loss: {loss:.4f}")
        return self


        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def predict(self, x, preprocessed=False):
        """
        Ouput the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # depend on whether you processed the data
        if preprocessed == False:
            x,_=self._preprocessor(x,training=False)
            temp_x = x
            x_tensor = torch.from_numpy(temp_x).float()
            y_hat = self.model(x_tensor)
            print(y_hat)
        else:
            temp_x = x
            x_tensor = torch.from_numpy(temp_x).float()
            y_hat = self.model(x_tensor)

        return np.exp(y_hat.detach().numpy())

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y, preprocessed=False):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        predictions = self.predict(x, preprocessed=preprocessed)
        r2=r2_score(y.values, predictions)
        pre=[]
        ground_truth=[]

        ground_truth.extend(y.values)
        pre.extend(predictions)
        data_tuples = list(zip(ground_truth, pre))
        df=pd.DataFrame(data_tuples, columns=['ground_truth', 'predictions'])
        df.to_csv("result_predictions.csv")

        print(r2)
        return r2

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model):
    """
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor():
    """
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def RegressorHyperParameterSearch(x,y):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.

    Returns:
        The function should return your optimised hyper-parameters.

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    parameters = {'activations':[["relu", "relu","relu","identity"],["relu", "relu","identity"]],
                  'neurons': [[32,32,32,1],[32,32,1]]}
    clf = GridSearchCV(Regressor(x),cv=5,param_grid=parameters,scoring='r2',return_train_score=True)

    print("begin to find hyperparameter ============")
    clf.fit(x, y)
    # check which hyperparameters gave the best score
    print("=================Finally the parameter is================== :")
    print("best_params_",clf.best_params_)
    print("best_score",clf.best_score_)

    # examine cross-validation results
    print("cv_results_",clf.cv_results_)

    #store the intermediate result in the csv
    print(clf.cv_results_.keys())
    pd.DataFrame({'param': clf.cv_results_["params"],
                  'mean_test_score': clf.cv_results_["mean_test_score"],
                  'mean_train_score': clf.cv_results_["mean_train_score"],
                  'rank_test_score': clf.cv_results_["rank_test_score"],
                  }).to_csv("result.csv")

    return clf.best_estimator_["params"]
    # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv")

    # Spliting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    #Training
    # This example trains on the whole available dataset.
    # You probably want to separate some held-out data
    # to make sure the model isn't overfitting



    #split dataset into test and train
    xv = x_train.values
    yv = y_train.values
    x_train_s, x_test_s, y_train_s, y_test_s = train_test_split(
        xv, yv, test_size=0.2, random_state=0)
    x_train_s = pd.DataFrame(data=x_train_s, columns=[x_train.columns])
    x_test_s = pd.DataFrame(data=x_test_s, columns=[x_train.columns])
    y_train_s = pd.DataFrame(data=y_train_s, columns=[y_train.columns])
    y_test_s = pd.DataFrame(data=y_test_s, columns=[y_train.columns])


    #============tuned model================

    regressor = Regressor(x_train_s, nb_epoch=1500, batch_size=32, learning_rate=0.01, loss_fun="mse", neurons=[40,40,40,40,1],
                          activations=["relu", "relu","relu", "relu","identity"])
    regressor.fit(x_train_s, y_train_s)
    #save_regressor(regressor)

    #r2
    r2 = regressor.score(x_test_s, y_test_s)
    print("\ntuned Regressor score: {}\n".format(r2))



    #find tne hyperparameter
    #best = RegressorHyperParameterSearch(x_train, y_train)

    #==================initial model===============
    initial_regressor = Regressor(x_train_s, nb_epoch = 10)
    initial_regressor.fit(x_train_s, y_train_s)
    r2_initial = initial_regressor.score(x_test_s, y_test_s)
    print("\n initial Regressor score: {}\n".format(r2_initial))

    #===========================
if __name__ == "__main__":
    example_main()
