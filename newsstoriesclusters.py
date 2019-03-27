# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 18:56:55 2019

@author: Samy Abud Yoshima
"""
import math
import numpy as np
import pandas as pd
from scipy import stats
import sys
import os
import nltk
from nltk.tokenize import RegexpTokenizer, wordpunct_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer 
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cluster
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# total number of articles to process
N = 100
# in memory stores for the topics, titles and contents of the news stories
topics_array = []
titles_array = []
corpus = []
for i in range(0, N):
    # get the contents of the article
    with open("article-" + str(i) + ".txt", 'r',errors='ignore') as myfile:
        d1=myfile.read().replace('\n', '')
        d1 = d1.lower()
        corpus.append(d1)
    #get the original topic of the article
    with open("topic-" + str(i) + ".txt", 'r',encoding="utf8") as myfile:
        to1=myfile.read().replace('\n', '')
        to1 = to1.lower()
        topics_array.append(to1)
    #get the title of the article
    with open("title-" + str(i) + ".txt", 'r',encoding="utf8") as myfile:
        ti1=myfile.read().replace('\n', '')
        ti1 = ti1.lower()
        titles_array.append(ti1)

parent = os.path.dirname(os.path.realpath(__file__))
sys.path.append(parent+"C:\\Users\\____\\MITIE-master\\mitielib")
from mitie import *
from collections import defaultdict
print("loading NER model...")
ner = named_entity_extractor("C:\\Users\\____\\MITIE-master\\MITIE-models\\english\\ner_model.dat")
print("\nTags output by this NER model:", ner.get_possible_ner_tags())
#entity subset array
entity_text_array = []
for i in range(0, N):
    # Load the article contents text file and convert it into a list of words.
    tokens = tokenize(load_entire_file("article-" + str(i) + ".txt"))
    # extract all entities known to the ner model mentioned in this article
    entities = ner.extract_entities(tokens)
# extract the actual entity words and append to the array
    for e in entities:
        range_array = e[0]
        tag = e[1]
        score = e[2]
        score_text = "{:0.3f}".format(score)
        entity_text = " ".join(tokens[j].decode() for j in range_array)
        entity_text_array.append(entity_text.lower())

# remove duplicate entities detected
entity_text_array = np.unique(entity_text_array)
vect = TfidfVectorizer(sublinear_tf=True, max_df=0.5, analyzer='word',stop_words='english', vocabulary=entity_text_array)
corpus_tf_idf = vect.fit_transform(corpus)

# total number of clusters
n_clusters = len(np.unique(topics_array))
# change n_clusters to equal the number of clusters desired
n_components = n_clusters
#spectral clustering
spectral = cluster.SpectralClustering(n_clusters= n_clusters,
eigen_solver='arpack',
affinity="nearest_neighbors",
n_neighbors = 10, 
random_state = n_clusters)
spectral.fit(corpus_tf_idf)

# Output format (article_number, topic, spectral_clustering_cluster_number, article_title)
if hasattr(spectral, 'labels_'):
    labels = spectral.labels_.astype(np.int)
    
for i in range(0, len(labels)):
    print(i, topics_array[i], labels [i], titles_array[i])
print(spectral.fit_predict)
print(spectral.affinity_matrix_)

TOPICS = np.unique(topics_array)
TOPICN = [n for n in range(0,len(TOPICS))]
print(TOPICS,TOPICN)
for t in TOPICS:
    i = list(TOPICS).index(t)
    elem = [list(corpus).index(n) for n,cl in zip(corpus,topics_array) if cl == TOPICS[i] ]
    globals()['S'+str(i)] = elem
    globals()['ones'+str(i)] = np.ones((len(globals()['S'+str(i)]),),dtype=int).tolist()
    globals()['SS'+str(i)] = list(zip(globals()['S'+str(i)],globals()['ones'+str(i)]))
    

fig, ax = plt.subplots()
#cnames = ['blue','red','green','black','yellow','pink','purple']
cnames = ['k','r','g','b','m','c','y']
for i in TOPICN:
    ax.broken_barh(globals()['SS'+str(i)],(i,1), facecolors = cnames[i])
ax.set_yticklabels(TOPICS)
ax.set_xlim(0,N)
ax.set_ylim(0,len(TOPICS))
ax.set_xlabel('Model Prediction')
ax.set_title('Model predicition versus TAG')
ax.grid(True)
fig.savefig("model eval.png")   
plt.show()


from scipy.sparse import csr_matrix
data =csr_matrix(corpus_tf_idf).toarray()
data = data*100
mea = data.mean()
std= data.std()
datan = ((data - mea) / std)
cov = np.cov(datan)
eig_val, eig_vec = np.linalg.eig(cov) # eigenvectors and eigenvalues from the cov matrix
eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs.sort(key=lambda x: x[0], reverse=True)
matrix_w = np.hstack((eig_pairs[1][1].reshape(len(corpus),1), eig_pairs[2][1].reshape(len(corpus),1)))
Y = matrix_w.T.dot(datan)

fig=plt.figure(3)
n_cl=n_clusters
X =  pd.DataFrame(data = Y.T, columns = ['PC2', 'PC3'])
kmeans = KMeans(n_clusters=n_cl, random_state=0)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
labels = np.unique(kmeans.labels_)
newcmp = ListedColormap(cnames)
#labels = [i for i in range(0,n_cl)]
plt.scatter(X['PC2'], X['PC3'], c=y_kmeans, s=20, cmap=newcmp)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=50, alpha=1);
for label, x, y in zip(TOPICS, centers[:, 0], centers[:, 1]):
    plt.annotate(label,xy=(x, y),xytext=(20, 20),
        textcoords='offset points', ha='center', va='bottom',
        bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.25),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
plt.xlabel('PC2')
plt.ylabel('PC3')
plt.grid(True)
fig.savefig("clusters in PC2xPC3.png")   
plt.show()






