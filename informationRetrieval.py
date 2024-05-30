from util import *
from collections import defaultdict  # Importing defaultdict from collections module
import numpy as np  # Importing numpy library and aliasing it as np

class InformationRetrieval():  # Defining a class named InformationRetrieval

    def __init__(self):  # Initializing the class with constructor
        self.index = None  # Initializing index attribute to None

    def buildIndex(self, docs, docIDs):  # Method to build document index
        """
        Builds the document index in terms of the document
        IDs and stores it in the 'index' class variable

        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is
            a document and each sub-sub-list is a sentence of the document
        arg2 : list
            A list of integers denoting IDs of the documents
        Returns
        -------
        None
        """

        index = None  # Initializing index variable to None

        #Fill in code here

        index = {tokens: [] for d in docs for sentence in d for tokens in sentence}  # Creating a dictionary with tokens as keys and empty lists as values
        for i in range(len(docs)):  # Iterating through each document
            doc = [token for sent in docs[i] for token in sent]  # Flattening the document into a list of tokens
            for j in docs[i]:  # Iterating through each sentence in the document
                for k in j:  # Iterating through each token in the sentence
                    if k in index.keys():  # Checking if the token is in the index keys
                        if [docIDs[i], doc.count(k)] not in index[k]:  # Checking if the document ID and token count pair is not already in the index
                            index[k].append([docIDs[i], doc.count(k)])  # Appending the document ID and token count pair to the index
        self.index = (index, len(docs), docIDs)  # Storing the index along with the number of documents and document IDs


    def rank(self, queries):  # Method to rank documents according to relevance for each query
        """
        Rank the documents according to relevance for each query

        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is a query and
            each sub-sub-list is a sentence of the query
        

        Returns
        -------
        list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        """

        doc_IDs_ordered = []  # Initializing a list to store ordered document IDs for each query

        #Fill in code here
        index, doc_num , doc_ID = self.index  # Retrieving index, number of documents, and document IDs

        D = np.zeros((doc_num, len(index.keys())))  # Creating a matrix to store document-term frequencies
        key = list(index.keys())  # Getting list of keys from index dictionary

        for i in range(len(key)):  
            for l in index[key[i]]:  # Iterating through each document-term frequency pair for the term
                D[l[0]-1, i] = l[1]  # Storing the document-term frequency in the matrix

        idf = np.zeros((len(key), 1))  # Creating a matrix to store inverse document frequencies
        for i in range(len(key)):  
            idf[i] = np.log10(doc_num / (len(index[key[i]])))  # Calculating IDF for the term

        for i in range(D.shape[0]):  # Iterating through each document
            D[i, :] = D[i, :] * idf.T  # Scaling document-term frequencies by IDF

        for i in range(len(queries)):  # Iterating through each query
            query = defaultdict(list)  # Creating a defaultdict to store query terms and their frequencies
            for j in queries[i]:  # Iterating through each sentence in the query
                for k in j:  # Iterating through each term in the sentence
                    if k in index.keys():  # Checking if the term is in the index
                        query[k] = index[k]  # Adding the term and its document-term frequency pairs to the query dictionary
            query = dict(query)  # Converting query defaultdict to a regular dictionary
            Q = np.zeros((1, len(key)))  # Creating a matrix to represent the query
            for m in range(len(key)):  
                if key[m] in query.keys():  # Checking if the term is in the query
                    Q[0, m] = 1  # Setting the corresponding entry in the query matrix to 1

            Q = Q*idf.T  # Scaling query term frequencies by IDF

            # cosine similarity
            temp = []  # Initializing a list to store cosine similarity scores
            for d in range(D.shape[0]):  # Iterating through each document
                simi = np.dot(Q[0, :], D[d, :]) / ((np.linalg.norm(Q[0, :]) + 1e-4) * (np.linalg.norm(D[d, :]) + 1e-4))  # Calculating cosine similarity between query and document
                temp.append(simi)  # Appending the similarity score to the list
            doc_IDs_ordered .append([x for _, x in sorted(zip(temp, doc_ID), reverse=True)])  # Sorting document IDs based on similarity scores and appending to the result list
        return doc_IDs_ordered  # Returning the ordered document IDs for each query
