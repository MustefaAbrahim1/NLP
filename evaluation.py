from util import *
import matplotlib.pyplot as plt


class Evaluation():

    def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of precision of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The precision value as a number between 0 and 1
        """

        precision = -1  # Initializing precision variable

        #Fill in code here
        relevance = np.zeros((len(query_doc_IDs_ordered),1))  # Creating an array to store relevance of each document
        for i in range(len(query_doc_IDs_ordered)):
            if query_doc_IDs_ordered[i] in true_doc_IDs:
                relevance[i] = 1  # Setting relevance to 1 for relevant documents

        precision = relevance[:k].sum()/k  # Calculating precision as the number of relevant documents at top-k divided by k

        return precision


    def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of precision of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries for which the documents are ordered
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The mean precision value as a number between 0 and 1
        """

        meanPrecision = 0  # Initializing meanPrecision variable

        #Fill in code here
        qrels_df = pd.DataFrame(qrels)  # Converting qrels to DataFrame
        for i in range(len(query_ids)):
            query_doc_IDs_ordered = doc_IDs_ordered[i]  # Getting ordered document IDs for the current query
            query_id = query_ids[i]  # Getting current query ID
            true_doc_IDs = list(map(int,list(qrels_df[qrels_df['query_num'] == str(query_id)]['id'])))  # Getting true relevant document IDs for the current query
            meanPrecision += self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)  # Adding precision for the current query to the mean precision
        meanPrecision /= len(query_ids)  # Calculating mean precision

        return meanPrecision

	
    def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of recall of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The recall value as a number between 0 and 1
        """

        recall = -1  # Initializing recall variable

        #Fill in code here
        relevance = np.zeros((len(query_doc_IDs_ordered),1))  # Creating an array to store relevance of each document
        for i in range(len(query_doc_IDs_ordered)):
            if query_doc_IDs_ordered[i] in true_doc_IDs:
                relevance[i] = 1  # Setting relevance to 1 for relevant documents

        recall = relevance[:k].sum()/len(true_doc_IDs)  # Calculating recall as the number of relevant documents at top-k divided by total relevant documents

        return recall


    def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of recall of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries for which the documents are ordered
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The mean recall value as a number between 0 and 1
        """

        meanRecall = 0  # Initializing meanRecall variable

        #Fill in code here
        qrels_df = pd.DataFrame(qrels)  # Converting qrels to DataFrame
        for i in range(len(query_ids)):
            query_doc_IDs_ordered = doc_IDs_ordered[i]  # Getting ordered document IDs for the current query
            query_id = query_ids[i]  # Getting current query ID
            true_doc_IDs = list(map(int,list(qrels_df[qrels_df['query_num'] == str(query_id)]['id'])))  # Getting true relevant document IDs for the current query
            meanRecall += self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)  # Adding recall for the current query to the mean recall
        meanRecall /= len(query_ids)  # Calculating mean recall

        return meanRecall


    def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of fscore of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The fscore value as a number between 0 and 1
        """

        fscore = -1  # Initializing fscore variable

        #Fill in code here
        precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)  # Calculating precision for the current query
        recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)  # Calculating recall for the current query

        if precision == 0 or recall == 0:  # Checking for division by zero
            fscore = 0
        else:
            fscore = (2 * precision * recall) / (precision + recall)  # Calculating fscore using precision and recall

        return fscore


    def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of fscore of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries for which the documents are ordered
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value
        
        Returns
        -------
        float
            The mean fscore value as a number between 0 and 1
        """

        meanFscore = 0  # Initializing meanFscore variable

        #Fill in code here
        qrels_df = pd.DataFrame(qrels)  # Converting qrels to DataFrame
        for i in range(len(query_ids)):
            query_doc_IDs_ordered = doc_IDs_ordered[i]  # Getting ordered document IDs for the current query
            query_id = query_ids[i]  # Getting current query ID
            true_doc_IDs = list(map(int,list(qrels_df[qrels_df['query_num'] == str(query_id)]['id'])))  # Getting true relevant document IDs for the current query
            meanFscore += self.queryFscore(query_doc_IDs_ordered, query_id, true_doc_IDs, k)  # Adding fscore for the current query to the mean fscore
        meanFscore /= len(query_ids)  # Calculating mean fscore

        return meanFscore
	

    def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of nDCG of the Information Retrieval System
        at given value of k for a single query

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of IDs of documents relevant to the query (ground truth) # some things wrong here, it should be having relevance rating also, so it must be a dict or df
        arg4 : int
            The k value

        Returns
        -------
        float
            The nDCG value as a number between 0 and 1
        """

        nDCG = 0  # Initializing nDCG variable

        #Fill in code here
        relevance = np.zeros((len(query_doc_IDs_ordered), 1))  # Creating an array to store relevance of each document
        true_doc_IDs["position"] = 4 - true_doc_IDs["position"]  # Reversing the relevance ratings to match the relevance scale

        # Finding ideal DCG value
        true_doc_IDs_sorted = true_doc_IDs.sort_values("position", ascending=False)  # Sorting true_doc_IDs by relevance rating
        DCG_ideal = true_doc_IDs_sorted.iloc[0]["position"]  # Taking the relevance rating of the top document as the first value of DCG_ideal
        for i in range(1, min(k, len(true_doc_IDs))):  # Calculating ideal DCG for top-k documents
            DCG_ideal += true_doc_IDs_sorted.iloc[i]["position"] * np.log(2) / np.log(i + 1)

        t_doc_IDs = list(map(int, true_doc_IDs["id"]))  # Extracting relevant document IDs
        for i in range(k):
            if query_doc_IDs_ordered[i] in t_doc_IDs:
                relevance[i] = true_doc_IDs[true_doc_IDs["id"] == str(query_doc_IDs_ordered[i])].iloc[0]["position"]  # Setting relevance of retrieved documents

        for i in range(k):
            nDCG += relevance[i] * np.log(2) / np.log(i + 2)  # Calculating nDCG

        nDCG = nDCG / DCG_ideal  # Normalizing nDCG

        return nDCG


    def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of nDCG of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries for which the documents are ordered
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The mean nDCG value as a number between 0 and 1
        """

        meanNDCG = 0  # Initializing meanNDCG variable

        #Fill in code here
        qrels_df = pd.DataFrame(qrels)  # Converting qrels to DataFrame
        for i in range(len(query_ids)):
            query_doc_IDs_ordered = doc_IDs_ordered[i]  # Getting ordered document IDs for the current query
            query_id = query_ids[i]  # Getting current query ID
            true_doc_IDs = qrels_df[["position","id"]][qrels_df["query_num"] == str(query_id)]  # Getting true relevant document IDs and relevance ratings for the current query
            meanNDCG += self.queryNDCG(query_doc_IDs_ordered, query_id, true_doc_IDs, k)  # Adding nDCG for the current query to the mean nDCG
        meanNDCG /= len(query_ids)  # Calculating mean nDCG

        return meanNDCG


    def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of average precision of the Information Retrieval System
        at a given value of k for a single query (the average of precision@i
        values for i such that the ith document is truly relevant)

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The average precision value as a number between 0 and 1
        """

        avgPrecision = 0  # Initializing avgPrecision variable

        #Fill in code here
        relevance = np.zeros((len(query_doc_IDs_ordered), 1))  # Creating an array to store relevance of each document
        for i in range(len(query_doc_IDs_ordered)):
            if query_doc_IDs_ordered[i] in true_doc_IDs:
                relevance[i] = 1  # Setting relevance to 1 for relevant documents

        for i in range(min(k, len(relevance))):
            if relevance[i] == 1:
                avgPrecision += self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, i + 1)  # Calculating precision@i and adding to avgPrecision

        if np.sum(relevance[:k]) == 0:  # Handling division by zero
            return 0
        else:
            return avgPrecision / np.sum(relevance[:k])  # Calculating average precision


    def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
        """
        Computation of MAP of the Information Retrieval System
        at given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The MAP value as a number between 0 and 1
        """

        meanAveragePrecision = 0  # Initializing meanAveragePrecision variable

        #Fill in code here
        qrels_df = pd.DataFrame(q_rels)  # Converting q_rels to DataFrame
        for i in range(len(query_ids)):
            query_doc_IDs_ordered = doc_IDs_ordered[i]  # Getting ordered document IDs for the current query
            query_id = query_ids[i]  # Getting current query ID
            true_doc_IDs = list(map(int, list(qrels_df[qrels_df['query_num'] == str(query_id)]['id'])))  # Getting true relevant document IDs for the current query
            meanAveragePrecision += self.queryAveragePrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)  # Adding average precision for the current query to the mean average precision
        meanAveragePrecision /= len(query_ids)  # Calculating mean average precision

        return meanAveragePrecision
    
    
# This is to plot all the metrics values at different k
evaluator = Evaluation()
def Evaluation_metrics(doc_IDs_ordered, query_ids, qrels, n_comp, op_folder = './',save_results = 2, verbose = 1, title_name = " "):
    """
    doc_IDs_ordered: List, the order of the retrieved docs by our model

    query_ids: List, values from 1 to 225. [1,2,3,..., 225]

    qrels: List, relevant documents for each query(/cranfield/cran_qrels.json)

    n_comp: integer, this argument used by the LSA model. Number of components considered.

    op_folder: Output Folder path, this is the folder path to save the results

    save_results : 0    ===> Output only the results table, not the plots
                 : 1    ===> Plots + Table Results
    title_name: str, title name of the plot(applicable when save_results = 1)

    """
    precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
    for k in range(1,11):
        precision = evaluator.meanPrecision(
            doc_IDs_ordered, query_ids, qrels, k)
        precisions.append(precision)
        recall = evaluator.meanRecall(
            doc_IDs_ordered, query_ids, qrels, k)
        recalls.append(recall)
        fscore = evaluator.meanFscore(
            doc_IDs_ordered, query_ids, qrels, k)
        fscores.append(fscore)

        MAP = evaluator.meanAveragePrecision(
            doc_IDs_ordered, query_ids, qrels, k)
        MAPs.append(MAP)
        nDCG = evaluator.meanNDCG(
            doc_IDs_ordered, query_ids, qrels, k)
        nDCGs.append(nDCG)
        if (verbose):
            print("Precision, Recall and F-score @ " +  
                str(k) + " : " + str(precision) + ", " + str(recall) + 
                ", " + str(fscore))
            print("MAP, nDCG @ " +  
                str(k) + " : " + str(MAP) + ", " + str(nDCG))
        # if (save_results > 0):
        # # saving the results
        #     with open(op_folder+'Results/LSA_'+str(n_comp)+'.txt', 'a') as f:
        #         f.write(str(k) + " , " + str(precision) + ", " + str(recall) + 
        #                 ", " + str(fscore)+", "+str(MAP) + ", " + str(nDCG)+'\n')
        #     with open(op_folder+'Results/metrics_'+str(k)+'.txt', 'a') as f:
        #         f.write(str(n_comp) + " , " + str(precision) + ", " + str(recall) + 
        #                 ", " + str(fscore)+", "+str(MAP) + ", " + str(nDCG)+'\n')
            
    # Plot the metrics and save plot 
    if (save_results == 1):
        plt.figure(figsize=(10,5))
        plt.plot(range(1, 11), precisions, label="Precision")
        plt.plot(range(1, 11), recalls, label="Recall")
        plt.plot(range(1, 11), fscores, label="F-Score")
        plt.plot(range(1, 11), MAPs, label="MAP")
        plt.plot(range(1, 11), nDCGs, label="nDCG")
        plt.legend()
        plt.title(title_name)
        plt.xlabel("k")
        #plt.savefig(op_folder + "Plots/LSA_"+str(n_comp)+".png")
    return