import re
import time

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm


def pre_process_data(df):
    corpus = []
    print('--Data Cleaning in progress, please wait..--')
    time.sleep(1)
    for i in tqdm(range(1, len(df.values))):
        content = remove_special(df['content'][i])
        title = remove_special(df['title'][i])
        content = to_lower(content)
        title = to_lower(title)
        content = stem_stopword(content)
        title = stem_stopword(title)
        paragraph = content + title
        corpus.append(paragraph)
        # print(i)
    return corpus


def pre_process_article(data):
    data = remove_special(data)
    data = to_lower(data)
    data = stem_stopword(data)
    return data


def remove_special(data):
    return re.sub('[^a-zA-Z]', ' ', data)


def to_lower(data):
    return data.lower()


def stem_stopword(data):
    wordnet_lemmatizer = WordNetLemmatizer()
    data = data.split()
    data = [wordnet_lemmatizer.lemmatize(word) for word in data if
            not word in set(stopwords.words('english')) and len(word) != 1]
    data = ' '.join(data)
    return data
