from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def transform_data(docs):
  """Creates LSA vectors for machine learning."""
  vect = TfidfVectorizer()
  dtm = vect.fit_transform(docs)

  # Calculate number of features for minimum components
  n_features = dtm.get_shape()[1]
  n_components = 100 if n_features > 100 else n_features - 1

  # Run SVD for LSA Vectors
  svd = TruncatedSVD(n_components=n_components)
  vectors = svd.fit_transform(dtm)
  return vect, svd, vectors

def transform_new(vectorizer, svd, doc):
  """Creates LSA vectors from string object"""
  sparse_vectors = vectorizer.transform(doc)
  lsa_vectors = svd.transform(sparse_vectors)
  return lsa_vectors
