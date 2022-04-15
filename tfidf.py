import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
# import stopwords

class preproc:
  def __init__(self, df):
      self.df = df
      for i in range(len(self.df['Plot'])):
        self.df.at[i,'Plot'] = self.preprocess(self.df.at[i,'Plot'])
      

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


class tf_idf:
    def __init__(self, df):
        self.df = df     

    def create_tfidf_features(self, col, max_features=5000, max_df=0.95, min_df=2):
        """ Creates a tf-idf matrix for the `corpus` using sklearn. """
        corpus = self.df[col].tolist()
        tfidf_vectorizor = TfidfVectorizer(decode_error='replace', strip_accents='unicode', analyzer='word',
                                        stop_words='english', ngram_range=(1, 1), max_features=max_features,
                                        norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True,
                                        max_df=max_df, min_df=min_df)

        X = tfidf_vectorizor.fit_transform(corpus)
        #   arr = tfidf_vectorizor.get_feature_names_out()
        #   print(arr)
        #   print(len(arr))
        #   print(X)
        print('tfidf matrix successfully created.')
            
        #   return X, tfidf_vectorizor
        with open('x.pickle', 'wb') as handle:
            pickle.dump(X, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open('vectorizor.pickle', 'wb') as handle:
            pickle.dump(tfidf_vectorizor, handle, protocol=pickle.HIGHEST_PROTOCOL)


df = pd.read_csv(r'preprocessed_dataset.csv')
# p = preproc(df)
# df = p.df

t = tf_idf(df)
t.create_tfidf_features('Plot')