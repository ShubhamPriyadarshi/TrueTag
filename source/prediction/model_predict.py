import os
import sys

from newspaper import Article

sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'training'))

from tfidf import tf_idf
from cleaning import pre_process_article


def predict(n, url):
    article = Article(url, language="en")
    article.download()
    article.parse()
    article.nlp()
    content = pre_process_article(str(article.text))
    title = pre_process_article(str(article.title))
    tf_idf(content, title, n)
