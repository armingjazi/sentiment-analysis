import numpy as np

from trading_sentiment_analysis.train.activation import sigmoid
from trading_sentiment_analysis.train.cost import batch_gradient_descent
from trading_sentiment_analysis.train.feature import extract_features_idf
from trading_sentiment_analysis.train.normalization import normalize, normalize_with_mean_var


class RegressionModel:
    def __init__(self, shape=(3, 1)):
        self.weights = np.zeros(shape)  # Initialize weights to zero
        self.mean = None
        self.var = None

    def predict(self, inputs):
        """
        Predict using the learned weights.
        :param inputs: Feature matrix (already TF-IDF processed)
        :return: Predicted probabilities
        """
        z = np.dot(inputs, self.weights)
        return sigmoid(z)
    
    def predict_sentiments(self, headlines, frequencies, idf_scores):
        """
        Predict using the learned weights.
        :param headlines: List of headlines
        :param frequencies: Frequencies of words
        :param idf_scores: IDF scores of words
        :return: Predicted probabilities
        """
        n = len(headlines)
        predict_input = np.zeros((n, 3))

        for i in range(n):
            predict_input[i, :] = extract_features_idf(headlines[i], frequencies, idf_scores)

        predict_input[:, 1:] = normalize_with_mean_var(predict_input[:, 1:], self.mean, self.var)

        return self.predict(predict_input)    

    def predict_sentiment(self, headline, frequencies, idf_scores):
        """
        Make a prediction using the model
        """
        features = extract_features_idf(headline, frequencies, idf_scores)
        return self.predict(features)
    
    def train(self, train_x, train_y, frequencies, idf_scores, learning_rate, iterations, batch_size=2048, optimizer=batch_gradient_descent):
        train_input = np.zeros((len(train_x), 3))

        print("Extracting features from training data...")
        
        for i in range(len(train_x)):
            train_input[i, :] = extract_features_idf(train_x[i], frequencies, idf_scores)

        train_input[:, 1:], self.mean, self.var = normalize(train_input[:, 1:])

        # labels
        train_target = train_y.reshape(-1, 1)

        print(f"Training input: \n {train_input[:5]} \n {train_input[-5:]} ")
        print(f"Training label: \n {train_target[:5]} \n {train_target[-5:]} ")

        costs, weights = optimizer(train_input, train_target, self.weights, learning_rate, iterations, batch_size)

        self.weights = weights

        return costs, weights
    
    def load(self, model_file, mean_var_file):
        """
        Load the model weights from a file
        """
        self.weights = np.load(model_file, allow_pickle=True)
        mean_var = np.load(mean_var_file, allow_pickle=True).item()
        self.mean = mean_var['mean']
        self.var = mean_var['var']
        if self.mean is None or self.var is None:
            raise ValueError("Mean and variance must be loaded from the file.")
        if self.weights is None:
            raise ValueError("Weights must be loaded from the file.")
        
        print(f"Model loaded from {model_file}")

    def save(self, model_file, mean_var_file):
        """
        Save the model weights to a file
        """
        np.save(model_file, self.weights)
        np.save(mean_var_file, {'mean': self.mean, 'var': self.var})
        
        print(f"Model saved to {model_file}")