from sklearn.base import TransformerMixin
from gensim.models import FastText
import numpy as np

class FastTextVectorizer(TransformerMixin):
    def __init__(self, vector_size=200, window=5, min_n = 3, max_n = 6, sg=1, workers=-1, epochs=10):
        self.vector_size = vector_size
        self.window = window
        self.min_n = min_n
        self.max_n = max_n
        self.sg = sg   
        self.workers = workers
        self.epochs = epochs  
        self.model = None

    def fit(self, X, y=None):
        sentences = [text.lower().split() for text in X]
        self.model = FastText(
            vector_size=self.vector_size,
            window=self.window,
            sg=self.sg,
            workers=self.workers,
            min_n=self.min_n,
            max_n=self.max_n

        )
        self.model.build_vocab(sentences)
        self.model.train(sentences, total_examples=len(sentences), epochs=self.epochs)
        return self
    
    def transform(self, X):
        vectors = []
        for text in X:
            words = text.lower().split()
            words_vec = [self.model.wv[word] for word in words if word in self.model.wv]
            vectors.append(np.mean(words_vec, axis=0))

        return np.array(vectors)


