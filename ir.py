import pickle
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
df = pd.read_csv(r'preprocessed_dataset.csv')
# df = pd.read_csv(r'preprocessed_dataset.csv')
# df.drop(['Director', 'Cast', 'Wiki Page', 'Origin/Ethnicity','Genre','Title','Release Year'], axis=1, inplace=True)
# df




class query_preproc:
  def __init__(self, query):
    self.query = query
    self.query = str(self.preprocess(self.query))

  def remove_stop_words(self, data):
      stop_words = stopwords.words('english')
      words = word_tokenize(str(data))
      new_text = ""
      for w in words:
          if w not in stop_words:
              new_text = new_text + " " + w
      return np.char.strip(new_text)

  def convert_to_lower_case(self, data):
      for i in data:
        i = i.lower()
      return np.char.lower(data)

  def remove_punctuation(self, data):
      symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
      for i in range(len(symbols)):
          data = np.char.replace(data, symbols[i], ' ')
          data = np.char.replace(data, "  ", " ")
      data = np.char.replace(data, ',', '')
      return data

  def remove_apostrophe(self, data):
      return np.char.replace(data, "'", "")

  def remove_single_characters(self, data):
      words = word_tokenize(str(data))
      new_text = ""
      for w in words:
          if len(w) > 1:
              new_text = new_text + " " + w
      return np.char.strip(new_text)

  def stemming(self, data):
      stemmer= PorterStemmer()
      
      tokens = word_tokenize(str(data))
      new_text = ""
      for w in tokens:
          new_text = new_text + " " + stemmer.stem(w)
      return np.char.strip(new_text)

  def convert_numbers(self, data):
      data = np.char.replace(data, "0", " zero ")
      data = np.char.replace(data, "1", " one ")
      data = np.char.replace(data, "2", " two ")
      data = np.char.replace(data, "3", " three ")
      data = np.char.replace(data, "4", " four ")
      data = np.char.replace(data, "5", " five ")
      data = np.char.replace(data, "6", " six ")
      data = np.char.replace(data, "7", " seven ")
      data = np.char.replace(data, "8", " eight ")
      data = np.char.replace(data, "9", " nine ")
      return data

  def preprocess(self, data):     
      data = self.convert_to_lower_case(data)
      data = self.convert_numbers(data)
      data = self.remove_punctuation(data) #remove comma seperately
      data = self.remove_stop_words(data)
      data = self.remove_apostrophe(data)
      data = self.remove_single_characters(data)
      data = self.stemming(data)
      return str(data)




class query:
  def __init__(self):
      pass

  def calculate_similarity(self, X, vectorizor, query, top_k=10):
      """ Vectorizes the `query` via `vectorizor` and calculates the cosine similarity of
      the `query` and `X` (all the documents) and returns the `top_k` similar documents."""

      q = query_preproc(query)
      query = [q.query]

      # Vectorize the query to the same length as documents
      query_vec = vectorizor.transform(query)
      # Compute the cosine similarity between query_vec and all the documents
      cosine_similarities = cosine_similarity(X,query_vec).flatten()
      # Sort the similar documents from the most similar to less similar and return the indices
      most_similar_doc_indices = np.argsort(cosine_similarities, axis=0)[:-top_k-1:-1]
      return (most_similar_doc_indices, cosine_similarities)

#   def show_similar_documents(self, df, cosine_similarities, similar_doc_indices):
#       counter = 1
#       for index in similar_doc_indices:
#           print("Index", index)
#           print('Top-{}, Similarity = {}'.format(counter, cosine_similarities[index]))
#           print(df.iloc[index,:])
#           print()
#           counter += 1
  
  def query_similarity_ranked_docs(self, queries):
    output = {}
    x = None
    vectorizor = None
    with open('x.pickle', 'rb') as handle:
        x = pickle.load(handle)

    with open('vectorizor.pickle', 'rb') as handle:
        vectorizor = pickle.load(handle)
    

    for query in queries:
        q = [query]
        docs , simi = self.calculate_similarity(x, vectorizor, q, top_k = len(df))
        count = 0
        required_doc = 5
        # if( required_doc in docs):
        #     print("True")
        for i in docs:
            if(simi[i] <= 0.1):
                continue
            if(query in output) :
                output[query] += 1
                # output[query][i] = simi[i]

            else:
                output[query] = 1
                # output[query] = {}
    return output

# q = query()
# docs , simi = q.calculate_similarity(x, vectorizor, query1, top_k = len(df))
# print(docs)
# queries = ["haunted place","funny comedy", "superhero","saves people","Alice follows rabbit hole"]
# output = q.query_similarity_ranked_docs(queries)
# print(df.iloc[5,:])
# print(output)

