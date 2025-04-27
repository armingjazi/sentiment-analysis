import kagglehub

def download_financial_news_data():
    model_handle = "miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests"
    return kagglehub.dataset_download(model_handle)



## main function for script call
if __name__ == '__main__':
    path = download_financial_news_data()
    print("Downloaded financial news data to: ", path)
