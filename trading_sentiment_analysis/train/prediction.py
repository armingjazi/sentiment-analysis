from trading_sentiment_analysis.train.feature import extract_features


def predict_headline(headline, model, frequencies):
    """Predict the sentiment of a headline.
    Input:
        headline: a headlline sentence
        model: a trained model
        frequencies: a dictionary mapping each (word, sentiment) pair to its frequency
    Output:
        prediction: the predicted sentiment of the headline (1 for positive, 0 for negative)
    """
    # extract features from the headline
    x = extract_features(headline, frequencies)
    
    # make prediction using the model
    prediction = model.predict(x)
    
    return prediction[0]