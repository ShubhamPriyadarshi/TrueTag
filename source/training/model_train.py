import os
import pickle
import sys

sys.path.append(os.path.join(sys.path[0], 'training'))
from cleaning import pre_process_data
import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

corpus_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'model',
                            'preprocessed_data.pkl')

def idf(data):
    print('--Calculating IDF values, please wait..--')
    cv = CountVectorizer()
    word_count_vector = cv.fit_transform(data)
    tf_idf_transformer = TfidfTransformer(norm='l1',smooth_idf=True, use_idf=True)
    tf_idf_transformer.fit(word_count_vector)
    idf_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'model',
                            'idf_data.pkl')
    tf_file = open(idf_path, 'wb')
    pickle.dump(cv, tf_file)
    pickle.dump(tf_idf_transformer, tf_file)


def cleaning(dataset, trunc=-1):
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'dataset',
                             dataset)
    articles1 = pd.read_csv(data_path)
    if trunc != -1:
        articles1 = articles1.truncate(after=trunc, axis='rows')

    corpus = pre_process_data(articles1)

    corpus_file = open(corpus_path, 'wb')
    pickle.dump(corpus, corpus_file)


def train(nc, dataset_name='articles.csv', trunc=-1):
    if nc is False:

        cleaning(dataset_name, trunc)
        file = open(corpus_path, 'rb')
        data = pickle.load(file)
        idf(data)
    elif nc is True:

        file = open(corpus_path, 'rb')
        data = pickle.load(file)
        idf(data)
