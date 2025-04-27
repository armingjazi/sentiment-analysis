import argparse
import numpy as np

from trading_sentiment_analysis.train.model import RegressionModel

def main():
    
    parser = argparse.ArgumentParser(description='predict')
    parser.add_argument('--model_file', type=str, nargs='?', default='model_weights.npy')
    parser.add_argument('--frequencies', type=str, nargs='?', default='frequencies.npy')
    parser.add_argument('--idf_scores', type=str, nargs='?', default='idf_scores.npy')
    parser.add_argument('--mean_var_file', type=str, nargs='?', default='mean_var.npy')
    parser.add_argument('--headline', type=str, nargs='?', default='Apple Inc. stock price prediction')
    
    model = RegressionModel()

    print("loading model...")

    args = parser.parse_args()
    model_file = args.model_file
    mean_var_file = args.mean_var_file
    model.load(model_file, mean_var_file)
        
    print("Model loaded.")
    
    print("loading frequencies...")
    frequencies = np.load(args.frequencies, allow_pickle=True).item()
    print("Frequencies loaded.")

    print("loading idf scores...")
    idf_scores = np.load(args.idf_scores, allow_pickle=True).item()
    print("IDF scores loaded.")

    prediction = model.predict_sentiment(args.headline, frequencies, idf_scores)

    label = 1 if prediction >= 0.5 else 0
    
    print(f"Prediction for headline '{args.headline}': {prediction}, Label: {label}")

    

if __name__ == '__main__':
    main()