from gensim.models import Word2Vec
import numpy as np
from sklearn.linear_model import SGDRegressor

class MultilingualW2V:

    def __init__(self, lang_1_code, lang_2_code, lang_1_index, lang_2_index, bilingual_token_list):
        '''
        set the data for the MultilingualW2V obj
        :param lang_1_code: the language code of lang1 -- used for persistence
        :param lang_2_code: the language code of lang2 -- used for persistence
        :param lang_1_index: the filename for the lang1 w2v index
        :param lang_2_index: the filename for the lang2 w2v index
        :param bilingual_token_list: a list of tuples of (<lang_1_word>, <lang_2_word>)
        '''

        self.lang_1_code = lang_1_code
        self.lang_2_code = lang_2_code

        # note that here we assume you are loading w2v indices in the default binary format (not the gensim default)
        self.lang_1_w2v = Word2Vec.load_word2vec_format(lang_1_index, binary=True)
        self.lang_2_w2v = Word2Vec.load_word2vec_format(lang_2_index, binary=True)

        # filter our mappings to make sure they're in both indexes
        self.bilingual_mappings = [(s,t) for s,t in bilingual_token_list if s in self.lang_1_w2v.vocab
                                   and t in self.lang_2_w2v.vocab]

        # this will be None until the model is trained
        self.W = None

    def train(self):
        X_train = np.vstack([self.lang_1_w2v[p[0]] for p in self.bilingual_mappings])
        Z_train = np.vstack([self.lang_2_w2v[p[1]] for p in self.bilingual_mappings])

        # there's a trick here -- train each y column separately (as a separate SGD problem. They are all independent
        # so this should be ok
        Z_cols = [Z_train[:,i] for i in range(Z_train.shape[1])]

        # train a model for each row of W, and get the W coefficients
        trained_coef_rows = []
        for z in Z_cols:
            clf = SGDRegressor()
            clf.fit(X_train, z)
            trained_coef_rows.append(clf.coef_)

        # now stack all the rows together to reconstruct W
        self.W = np.vstack(trained_coef_rows)

    def multilingual_most_similar(self, lang_1_tok, cutoff=10):
        '''
        find the most similar lang2 tokens to lang_1
        :param lang_1_tok: a token in lang1
        :return: list of the most similar tokens in lang_2
        '''

        top_mappings = []
        if lang_1_tok in self.lang_1_w2v.vocab:
           lang_1_vec = self.lang_1_w2v[lang_1_tok]
           z_prime = self.W.dot(lang_1_vec)

           # find the most similar n tokens in the lang2 index
           # this uses the gensim internal representation to efficiently compute the distance to every token
           distances = np.dot(self.lang_2_w2v.syn0norm, z_prime)
           top = np.argsort(distances)[::-1]

           # retrieve the actual tokens from their indices
           top_lang2_matches = [(self.lang_2_w2v.index2word[sim], float(distances[sim])) for sim in top][:cutoff]
           top_mappings = top_lang2_matches

        return top_mappings



