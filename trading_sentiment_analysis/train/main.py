import argparse
import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt

from trading_sentiment_analysis.train.frequency import build_freqs_docs, compute_idf
from trading_sentiment_analysis.train.model import RegressionModel

def train_test_split(data_file, ratio=0.8):
    data = read_csv(data_file)
    
    all_positive = data[data['label'] == 1]
    all_negative = data[data['label'] == 0]

    positive_train = all_positive.sample(frac=ratio)
    negative_train = all_negative.sample(frac=ratio)

    positive_test = all_positive.drop(positive_train.index)
    negative_test = all_negative.drop(negative_train.index)

    train_data = pd.concat([positive_train, negative_train], ignore_index=True)
    test_data = pd.concat([positive_test, negative_test], ignore_index=True)

    train_y = np.append(np.ones((len(positive_train), 1)), np.zeros((len(negative_train), 1)))
    test_y = np.append(np.ones((len(positive_test), 1)), np.zeros((len(negative_test), 1)))

    return train_data, test_data, train_y, test_y
    


def main():
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('data_file', type=str, nargs='?', default='data/labeled/cleaned_news.csv')
    parser.add_argument('--tfidf', action='store_true', help='Use TF-IDF for feature extraction')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-3, help='Learning rate for training')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of iterations for training')
    parser.add_argument('--model_file', type=str, default='model_weights.npy', help='File to save the model weights')
    parser.add_argument('--mean_var_file', type=str, default='mean_var.npy', help='File to save the mean and variance for normalization')
    parser.add_argument('--frequencies', type=str, default='frequencies.npy', help='File to save the frequencies')
    parser.add_argument('--idf_scores', type=str, default='idf_scores.npy', help='File to save the IDF scores')
    args = parser.parse_args()
    data_file = args.data_file

    train_data, test_data, train_y, test_y = train_test_split(data_file)
    print(f"Train data: {len(train_data)} \n {train_data.head()} \n {train_data.tail()} ")
    print(f"Test data: {len(test_data)} \n {test_data.head()} \n {test_data.tail()} ")
    print(f"Train y: {len(train_y)} \n {train_y[:5]} \n {train_y[-5:]} ")
    print(f"Test y: {len(test_y)} \n {test_y[:5]} \n {test_y[-5:]} ")

    train_x = train_data['title'].values

    print("Building frequencies...")
    freqs, doc_freqs, total_docs = build_freqs_docs(train_x, train_y)

    print("Training model...")

    model = RegressionModel()

    if args.tfidf:
        idf_scores = compute_idf(doc_freqs, total_docs)
        costs, weights = model.train(train_x, train_y, freqs, idf_scores, learning_rate=args.learning_rate, iterations=args.iterations, batch_size=args.batch_size)
    else:
        costs, weights = model.train(train_x, train_y, freqs, {}, learning_rate=args.learning_rate, iterations=args.iterations, batch_size=args.batch_size)

    print("Model trained.")

    print("Evaluating model...")
    test_x = test_data['title'].values
    test_probs = np.zeros((len(test_x), 1))
    if args.tfidf:
        test_probs = model.predict_sentiments(test_x, freqs, idf_scores)
    else:
        test_probs = model.predict_sentiments(test_x, freqs, {})

    test_preds = (test_probs >= 0.5).astype(int)
    
    accuracy = np.mean(test_preds.flatten() == test_y)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    print("saving model...")

    # Save the model weights
    model_file = args.model_file
    mean_var_file = args.mean_var_file
    model.save(model_file, mean_var_file)

    # save frequencies and idf scores
    freqs_path = args.frequencies
    np.save(freqs_path, freqs)
    print(f"Frequencies saved to {freqs_path}")
    if args.tfidf:
        idf_path = args.idf_scores
        np.save(idf_path, idf_scores)
        print(f"IDF scores saved to {idf_path}")
    else:
        print("TF-IDF not used, no IDF scores saved.")
    

   
    plt.plot(costs)
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.title("Training Loss")
    plt.show()

    

if __name__ == '__main__':
    main()