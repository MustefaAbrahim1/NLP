# Add your import statements here
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
#import from nltk 
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
from scipy.special import stdtr
# import from our python files
from evaluation import Evaluation
from util import *

# Add any utility functions here

def build_word_index(docs,doc_ids):
    corpora = [] # a list of words
    #word_map = {}
    for doc in docs:
        for sent in doc:
            for word in sent:
                corpora.append(word)
    corpora = set(corpora)  # unique words
    word_map = {word : idx for idx,word in enumerate(set(corpora),0)} # for assigning a unique label to each word

    return word_map

# This transforms the count matrix to tfidf matrix
def TF_IDF(docs, doc_ids, word_map, normalize=True, is_queries=False):
    """
    Computes the TF-IDF representation of documents.

    Parameters:
    - docs: list of list of lists representing documents
    - doc_ids: list containing IDs of documents
    - word_map: mapping of words to indices
    - normalize: bool, indicates whether to normalize the TF-IDF features
    - is_queries: bool, True if processing queries

    Returns:
    - tf_idf: numpy array, TF-IDF representation of documents
    """
    # Number of unique words
    m = len(set(word_map))
    # Number of documents
    n = len(doc_ids)

    # Initialize TF matrix
    tf = np.zeros((m, n))

    # Fill TF matrix
    for i in range(n):
        for sent in docs[i]:
            for word in sent:
                try:
                    tf[word_map[word]][doc_ids[i] - 1] += 1
                except:
                    pass
    
    # Compute IDF
    idf = tf != 0
    idf = np.sum(idf, axis=1)
    idf = np.log10(n / idf).reshape(-1, 1)
    tf_idf = idf * tf
    
    # If processing queries, return TF matrix
    if is_queries:
        tf_idf = tf
    
    # Normalize TF-IDF features if necessary
    if normalize:
        epsilon = 1e-4
        return tf_idf / (np.linalg.norm(tf_idf, axis=0) + epsilon)
    
    return tf_idf


def plot_one(qrels, doc_IDs_ordered, queries, k, model_name = ' ', bin_size = 20):

    """
    Runs all the observations like distribution of the recall, precision, ndcg..

    Input Arguments:
    qrels : Relevant documents for each queries, we import from here -> "./cranfield/cran_qrels.json"
    doc_IDs_ordered : List of list, output is the list of the retrieved documents for each query
    queries : List, list of all the queries
    k : int, number of top retrieved to be considered. Usually ranges from 1 to 10.
    bin_size : int, number of bins for histogram plot
    """
    df = pd.DataFrame(qrels)
    evaluator = Evaluation()

    # to store lists 
    ones_precision = zeros_precision = ones_recall = zeros_recall = []
    q_precision =  q_recall = q_fscore = []

    for i in range(len(doc_IDs_ordered)):
        true_doc_ids = list(map(int, df[df['query_num'] == str(i+1)]['id'].tolist()))
        precision = evaluator.queryPrecision(doc_IDs_ordered[i], i+1, true_doc_ids, k)
        q_precision.append(precision)
        recall = evaluator.queryRecall(doc_IDs_ordered[i], i+1, true_doc_ids, k)
        q_recall.append(recall)
        fscore = evaluator.queryFscore(doc_IDs_ordered[i], i+1, true_doc_ids, k)
        q_fscore.append(fscore)

        if precision == 1:
            ones_precision.append({'q_id':i+1, 'query': queries[i],
            'rel_docs':true_doc_ids, 'ret_docs':doc_IDs_ordered[i][:10]})
        if precision == 0:
            zeros_precision.append({'q_id':i+1, 'query': queries[i],
            'rel_docs':true_doc_ids, 'ret_docs':doc_IDs_ordered[i][:10]})
            
        if recall == 1:
            ones_recall.append({'q_id':i+1, 'query': queries[i],
            'rel_docs':true_doc_ids, 'ret_docs':doc_IDs_ordered[i][:10]})
        if recall == 0:
            zeros_recall.append({'q_id':i+1, 'query': queries[i],
            'rel_docs':true_doc_ids, 'ret_docs':doc_IDs_ordered[i][:10]})

    # ndcg for each query calculation
    q_ndcg = []
    for i in range(len(doc_IDs_ordered)):
        true_doc_ndcg = df[df['query_num'] == str(i+1)][['position', 'id']]
        ndcg = evaluator.queryNDCG(doc_IDs_ordered[i], i+1, true_doc_ndcg, k)
        q_ndcg.append(list(ndcg)[0])

    # Precision graph
    x_label = 'Precision @ k = ' + str(k)
    plt.figure(figsize = (10,5))
    plt.xlabel(x_label)
    plt.ylabel('Number of Queries')
    plt.title('Precision Distribution for ' + model_name)
    sns.distplot(q_precision, hist=True, kde=True,color = 'brown',
                hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 4})
    plt.show()

    # Recall graph
    x_label = 'Recall @ k = ' + str(k)
    plt.figure(figsize = (10,5))
    plt.xlabel(x_label)
    plt.ylabel('Number of Queries')
    plt.title('Recall Distribution for ' + model_name)
    sns.distplot(q_recall, hist=True, kde=True,color = 'brown',
                hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 4})
    plt.show()

    # Fscore graph
    x_label = 'Fscore @ k = ' + str(k)
    plt.figure(figsize = (10,5))
    plt.xlabel(x_label)
    plt.ylabel('Number of Queries')
    plt.title('Fscore Distribution for ' + model_name)
    sns.distplot(q_fscore, hist=True, kde=True,color = 'brown',
                hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 4})
    plt.show()

    # nDCG graph
    x_label = 'nDCG @ k = ' + str(k)
    plt.figure(figsize = (10,5))
    plt.xlabel(x_label)
    plt.ylabel('Number of Queries')
    plt.title('nDCG Distribution for ' + model_name)
    sns.distplot(q_ndcg, hist=True, kde=True,color = 'brown',
                hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 4})
    plt.show()

    return q_precision, q_recall, q_fscore, q_ndcg

def plot_pair(q_precision_1, q_precision_2, q_recall_1, q_recall_2, q_fscore_1, q_fscore_2, q_ndcg_1, q_ndcg_2, k, model1_name = ' ', model2_name = ' '):

    # Precision graph for the Model1 and Model2
    
    x_label = 'Precision @ k = ' + str(k)
    plt.figure(figsize = (10,5))
    plt.xlabel(x_label)
    plt.ylabel('Number of Queries')
    plt.title('Precision Distribution for ' + model1_name + ' Vs ' + model2_name)
    sns.distplot(q_precision_1, hist=True, kde=True,color = 'green',
                hist_kws={'edgecolor':'red'},kde_kws={'linewidth': 4}, label = model1_name)
    sns.distplot(q_precision_2, hist=True, kde=True,color = 'brown',
                hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 4}, label = model2_name)
    plt.legend()
    plt.show()

    # Recall graph for the Model1 and Model2
    
    x_label = 'Recall @ k = ' + str(k)
    plt.figure(figsize = (10,5))
    plt.xlabel(x_label)
    plt.ylabel('Number of Queries')
    plt.title('Recall Distribution for ' + model1_name + ' Vs ' + model2_name)
    sns.distplot(q_recall_1, hist=True, kde=True,color = 'green',
                hist_kws={'edgecolor':'red'},kde_kws={'linewidth': 4}, label = model1_name)
    sns.distplot(q_recall_2, hist=True, kde=True,color = 'brown',
                hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 4}, label = model2_name)
    plt.legend()
    plt.show()

    # Fscore graph for the Model1 and Model2
    
    x_label = 'Fscore @ k = ' + str(k)
    plt.figure(figsize = (10,5))
    plt.xlabel(x_label)
    plt.ylabel('Number of Queries')
    sns.distplot(q_fscore_1, hist=True, kde=True,color = 'green',
                hist_kws={'edgecolor':'red'},kde_kws={'linewidth': 4}, label = model1_name)
    sns.distplot(q_fscore_2, hist=True, kde=True,color = 'brown',
                hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 4}, label = model2_name)
    plt.title('Fscore Distribution for ' + model1_name + ' Vs ' + model2_name)
    plt.legend()
    plt.show()

    # Precision graph for the Model1 and Model2
    
    x_label = 'nDCG @ k = ' + str(k)
    plt.figure(figsize = (10,5))
    plt.xlabel(x_label)
    plt.ylabel('Number of Queries')
    sns.distplot(q_ndcg_1, hist=True, kde=True,color = 'green',
                hist_kws={'edgecolor':'red'},kde_kws={'linewidth': 4}, label = model1_name)
    sns.distplot(q_ndcg_2, hist=True, kde=True,color = 'brown',
                hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 4}, label = model2_name)
    plt.title('nDCG Distribution for ' + model1_name + ' Vs ' + model2_name)
    plt.legend()
    plt.show()

    return
def hypothesis_test(sample_1, sample_2):

    """
    Two Sample t-test
    Inputs:
    sample_1 : List, sample 1 values and  assign to a array
    sample_2 : List, sample 2 values and assign to b array

    Output:
    t_test : float, t-value of hypothesis testing
    p_value : float, p-value
    s1_dof: degree of freedom of sample 1, n_s1 - size of sample 1
    s2_dof: degree of freedom of sample 2, n_s2 - size of sample 2
    """
    # sample 1 details for t-test
    s1 = np.array(sample_1)
    s1_var = s1.var(ddof=1)  # we use ddof=1 help us to calculate variance of sample 
    n_s1 = s1.size         # while by default ddof=0 means variance of population
    s1_dof = n_s1 - 1

    # sample 2 details for t-test
    s2 = np.array(sample_2)
    s2_var = s2.var(ddof=1)
    n_s2 = s2.size
    s2_dof = n_s2 - 1

    # calculate t - test using the formula.
    t_test = (s1.mean() - s2.mean()) / np.sqrt(s1_var/n_s1 + s2_var/n_s2)
    dof = ((s1_var**2/n_s1 + s2_var**2/n_s2)**2 )/ ( (s1_var**4/(n_s1**2 * s1_dof)) + ((s2_var**4/(n_s2**2 * s2_dof))) )
   

    p_value = 2*stdtr(dof, -np.abs(t_test))

    print("formula: t = %g  p = %g" % (t_test, p_value))
    return
