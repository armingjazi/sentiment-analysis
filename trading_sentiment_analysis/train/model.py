import numpy as np

from trading_sentiment_analysis.train.activation import sigmoid
from trading_sentiment_analysis.train.cost import batch_gradient_descent
from trading_sentiment_analysis.train.feature import extract_features_idf
from trading_sentiment_analysis.train.normalization import batch_norm


class RegressionModel:
    def __init__(self, shape=(3, 1)):
        self.weights = np.zeros(shape)  # Initialize weights to zero

    def predict(self, inputs):
        """
        Make a prediction using the model
        :param inputs: Feature matrix
        :return: Predicted labels
        """
        return sigmoid(np.dot(inputs, self.weights))
    
    def train(self, train_x, train_y, frequencies, idf_scores, learning_rate, iterations, batch_size=2048, optimizer=batch_gradient_descent):
        train_input = np.zeros((len(train_x), 3))

        print("Extracting features from training data...")
        
        for i in range(len(train_x)):
            train_input[i, :] = extract_features_idf(train_x[i], frequencies, idf_scores)

        train_input[:, 1:] = batch_norm(train_input[:, 1:])

        # labels
        train_target = train_y.reshape(-1, 1)

        print(f"Training input: \n {train_input[:5]} \n {train_input[-5:]} ")
        print(f"Training label: \n {train_target[:5]} \n {train_target[-5:]} ")

        costs, weights = optimizer(train_input, train_target, self.weights, learning_rate, iterations, batch_size)

        self.weights = weights

        return costs, weights
