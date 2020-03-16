import os
import pickle


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn):
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results


def tf_idf(content, title, n):
    idf_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'model',
                            'idf_data.pkl')
    df_file = open(idf_path, 'rb')
    cv = pickle.load(df_file)
    tf_idf_transformer = pickle.load(df_file)
    feature_names = cv.get_feature_names()

    tf_idf_vector_content = tf_idf_transformer.transform(cv.transform([content]))
    tf_idf_vector_title = tf_idf_transformer.transform(cv.transform([title]))
    tf_idf_vector = tf_idf_vector_content + tf_idf_vector_title*0.75
    sorted_items = sort_coo(tf_idf_vector.tocoo())

    keywords = extract_topn_from_vector(feature_names, sorted_items, n)
    print(keywords)


"""
csv
exe
flask - API
"""
