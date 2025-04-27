import numpy as np

from trading_sentiment_analysis.train.activation import sigmoid

def cross_entropy_loss(target, prediction):
    """
    Compute the cross-entropy loss
    :param y: True labels
    :param y_hat: Predicted labels
    :return: Cross-entropy
    """
    # Clip predictions to avoid log(0)
    epsilon = 1e-15
    prediction = np.clip(prediction, epsilon, 1 - epsilon)
    clipped_one_minus_prediction = np.clip(1 - prediction, epsilon, 1 - epsilon)
    m = target.shape[0]
    
    return - 1 / m * (np.dot(target.T, np.log(prediction)) + np.dot((1 - target).T, np.log(clipped_one_minus_prediction)))

def cross_entropy_loss_gradient(inputs, targets, activations):
    """
    Compute the gradient of the cross-entropy loss
    :param inputs: Feature matrix
    :param targets: True labels
    :param activations: activations from the last layer
    :return: Gradient of the cross-entropy loss
    """
    m = inputs.shape[0]
    
    return np.dot(inputs.T, (activations - targets)) / m


def gradient_descent(inputs, targets, weights, learning_rate, iterations):
    """
    Perform gradient descent to learn theta
    :param inputs: Feature matrix
    :param targets: Target variable
    :param weights: Initial parameters
    :param learning_rate: Learning rate
    :param iterations: Number of iterations
    :return: Updated parameters
    """
    new_weights = weights
    for _ in range(0, iterations):

        print(f"Iteration {_ + 1} / {iterations}")

        layer_output = np.dot(inputs, weights)
        
        activations = sigmoid(layer_output)

        cost_function = cross_entropy_loss(targets, activations)

        new_weights = new_weights - (learning_rate) * cross_entropy_loss_gradient(inputs, targets, activations)

        print(f"\n Cost: {cost_function}")

    return float(cost_function), new_weights

def batch_gradient_descent(inputs, targets, weights, learning_rate, iterations, batch_size=1024):
    n = inputs.shape[0]
    new_weights = weights

    costs = []

    for epoch in range(iterations):

        permutation = np.random.permutation(n)
        inputs_shuffled = inputs[permutation]
        targets_shuffled = targets[permutation]

        epoch_cost = 0

        for i in range(0, n, batch_size):
            x_batch = inputs_shuffled[i:i+batch_size]
            y_batch = targets_shuffled[i:i+batch_size]

            layer_output = np.dot(x_batch, new_weights)
            activations = sigmoid(layer_output)
            cost = cross_entropy_loss(y_batch, activations)
            epoch_cost += float(cost) * x_batch.shape[0] # normalize by batch size

            new_weights = new_weights - learning_rate * cross_entropy_loss_gradient(x_batch, y_batch, activations)

        avg_epoch_cost = epoch_cost / n  
        costs.append(avg_epoch_cost)

        print(f"Iteration {epoch + 1} / {iterations} - Cost: {avg_epoch_cost:.6f}")

    return costs, new_weights