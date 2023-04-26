import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def calculate_tfidf(pdseries):
    """
    Calculates tfidf values for each word in a pandas series

    Inputs:
        pdseries (pandas series): a series of text
    
    Returns:
        tfIdf_df (pandas dataframe): a dataframe with the tfIdf value for each unique word
    """
    tfIdfVectorizer=TfidfVectorizer(use_idf=True)
    tfIdf_matrix = tfIdfVectorizer.fit_transform(pdseries)
    tfIdf_df = pd.DataFrame(tfIdf_matrix[0].T.todense(), index=tfIdfVectorizer.get_feature_names_out(), columns=["TF-IDF"])
    tfIdf_df = tfIdf_df.sort_values('TF-IDF', ascending=False)

    return tfIdf_df