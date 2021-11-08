import numpy as np
import nltk

nltk.download('twitter_samples')
from nltk.corpus import twitter_samples

from nltk.tokenize import TweetTokenizer
tt = TweetTokenizer()


def tokenizar(tweet):
    return tt.tokenize(tweet)

tweets = twitter_samples.strings('tweets.20150430-223406.json')
tokens = [tokenizar(t) for t in tweets]

t_limpo = [[i.lower() for i in msg if i.isalpha()] for msg in tokens]

# É necessário transformar as listas de tokens em strings para usar o scikit-learn
t_str = [' '.join(i) for i in t_limpo]


from sklearn.feature_extraction.text import TfidfVectorizer
TfidfVec = TfidfVectorizer(stop_words='english')

def similaridade(lista_str):
    tfidf = TfidfVec.fit_transform(lista_str)
    return (tfidf * tfidf.T).toarray()

sim = similaridade(t_str)
#print(sim)

lin = np.where((sim > 0.2) & (sim < 9))[0]
col = np.where((sim > 0.9) & (sim < 6))[1]
mat_sim = list(zip(lin, col))
mat_sim2 = {tuple(sorted(i)) for i in mat_sim}  # Transforma em tuplas e filtra duplicadas (sim(x, y) == sim(y, x))
mat_sim2 = list(mat_sim2)

# Conferindo com um exemplo:
#print(tweets[mat_sim2[0][0]], '\n', tweets[mat_sim2[0][1]])

print (tweets)
