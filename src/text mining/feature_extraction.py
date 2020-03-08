from __future__ import print_function, division


from sklearn.feature_extraction.text import HashingVectorizer,CountVectorizer,TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from helper.utilities import _randint, _randchoice, _randuniform

def TF():
    a, b = _randint(100, 1000), _randint(1, 10)
    vect= CountVectorizer(max_df=a, min_df=b)
    tmp = str(a) + "_" + str(b) + "_" + CountVectorizer.__name__
    return vect, tmp


def TFIDF():
    a, b = _randint(100, 1000), _randint(1, 10)
    c = _randchoice(['l1', 'l2', None])
    vect = TfidfVectorizer(max_df=a, min_df=b, norm=c)
    tmp = str(a) + "_" + str(b) + "_" + str(c) + "_" + TfidfVectorizer.__name__
    return vect, tmp

def HASHING():
    a = _randchoice([1000, 2000, 4000, 6000, 8000, 10000])
    b = _randchoice(['l1', 'l2', None])
    vect = HashingVectorizer(n_features=a, norm=b)
    tmp = str(a) + "_" + str(b) + "_" + HashingVectorizer.__name__
    return vect, tmp

def LDA_():
    a, b = _randint(100, 1000), _randint(1, 10)
    vect = CountVectorizer(max_df=a, min_df=b)
    a, b, c = _randint(10, 50), _randuniform(0, 1), _randuniform(0, 1)
    d, e, f = _randuniform(0.51, 1.0), _randuniform(1, 50), _randchoice([150,180,210,250,300])
    lda= LatentDirichletAllocation(n_components=a,doc_topic_prior=b, topic_word_prior=c,
                                   learning_decay=d,learning_offset=e,batch_size=f,
                                   max_iter=100,learning_method='online')
    tmp = str(a) + "_" + str(b) + "_" + str(c) + "_" + str(d) + "_" + str(e) + "_" + str(f) + "_" +LatentDirichletAllocation.__name__
    return [vect,lda], tmp


def extraction(text,vector):
    if type(vector) == list:
        corpus=vector[0].fit_transform(text).A
        return vector[1].fit_transform(corpus)
    else:
        data = vector.fit_transform(text)
        return data.A