from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from informationRetrievalLSA import InformationRetrieval 
from evaluation import Evaluation
from sys import version_info
import argparse
import json
import matplotlib.pyplot as plt

# Input compatibility for Python 2 and Python 3
if version_info.major == 3:
    pass
elif version_info.major == 2:
    try:
        input = raw_input
    except NameError:
        pass
else:
    print ("Unknown python version - input function not safe")


class SearchEngine:
# how the Search engine will work
    def __init__(self, args):
        self.args = args

        self.tokenizer = Tokenization() # first tokenize both query and docs
        self.sentenceSegmenter = SentenceSegmentation() # 2nd do sentences segmentation to each sentences
        self.inflectionReducer = InflectionReduction() # 3rd doing the stemming/lemmitazation
        self.stopwordRemover = StopwordRemoval() # 4th removing stop word 

        self.informationRetriever = InformationRetrieval() # 5th then retrieval of the relevent docs
        self.evaluator = Evaluation() # 6th checking the evaluation metrics for how much its relevant


    def segmentSentences(self, text): # sentence segmenter

        if self.args.segmenter == "naive":
            return self.sentenceSegmenter.naive(text)
        elif self.args.segmenter == "punkt":
            return self.sentenceSegmenter.punkt(text)

    def tokenize(self, text): #Call the required tokenizer

        if self.args.tokenizer == "naive":
            return self.tokenizer.naive(text)
        elif self.args.tokenizer == "ptb":
            return self.tokenizer.pennTreeBank(text)

    def reduceInflection(self, text):
        return self.inflectionReducer.reduce(text) # stemmer/lemmatizer 

    def removeStopwords(self, text):
        return self.stopwordRemover.fromList(text) #  stopword remover

  # Preprocess the queries - segment, tokenize, stem/lemmatize and remove stopwords
    def preprocessQueries(self, queries):
        # Segment queries
        segmentedQueries = []
        for query in queries:
            segmentedQuery = self.segmentSentences(query)
            segmentedQueries.append(segmentedQuery)
        json.dump(segmentedQueries, open(self.args.out_folder + "segmented_queries.txt", 'w'))
        # Tokenize queries
        tokenizedQueries = []
        for query in segmentedQueries:
            tokenizedQuery = self.tokenize(query)
            tokenizedQueries.append(tokenizedQuery)
        json.dump(tokenizedQueries, open(self.args.out_folder + "tokenized_queries.txt", 'w'))
        # Stem/Lemmatize queries
        reducedQueries = []
        for query in tokenizedQueries:
            reducedQuery = self.reduceInflection(query)
            reducedQueries.append(reducedQuery)
        json.dump(reducedQueries, open(self.args.out_folder + "reduced_queries.txt", 'w'))
        # Remove stopwords from queries
        stopwordRemovedQueries = []
        for query in reducedQueries:
            stopwordRemovedQuery = self.removeStopwords(query)
            stopwordRemovedQueries.append(stopwordRemovedQuery)
        json.dump(stopwordRemovedQueries, open(self.args.out_folder + "stopword_removed_queries.txt", 'w'))

        preprocessedQueries = stopwordRemovedQueries
        return preprocessedQueries

    def preprocessDocs(self, docs): # Preprocess the documents

        # Segment docs
        segmentedDocs = []
        for doc in docs:
            segmentedDoc = self.segmentSentences(doc)
            segmentedDocs.append(segmentedDoc)
        json.dump(segmentedDocs, open(self.args.out_folder + "segmented_docs.txt", 'w'))
        # Tokenize docs
        tokenizedDocs = []
        for doc in segmentedDocs:
            tokenizedDoc = self.tokenize(doc)
            tokenizedDocs.append(tokenizedDoc)
        json.dump(tokenizedDocs, open(self.args.out_folder + "tokenized_docs.txt", 'w'))
        # Stem/Lemmatize docs
        reducedDocs = []
        for doc in tokenizedDocs:
            reducedDoc = self.reduceInflection(doc)
            reducedDocs.append(reducedDoc)
        json.dump(reducedDocs, open(self.args.out_folder + "reduced_docs.txt", 'w'))
        # Remove stopwords from docs
        stopwordRemovedDocs = []
        for doc in reducedDocs:
            stopwordRemovedDoc = self.removeStopwords(doc)
            stopwordRemovedDocs.append(stopwordRemovedDoc)
        json.dump(stopwordRemovedDocs, open(self.args.out_folder + "stopword_removed_docs.txt", 'w'))

        preprocessedDocs = stopwordRemovedDocs
        return preprocessedDocs


    def evaluateDataset(self):
        """
        - Preprocesses queries and documents, storing them in the output folder. 
            - Calls the IR system. 
            - Evaluates precision, recall, fscore, nDCG, and MAP. 

        graphs of evaluation metrics are generated and saved to the output folder.

        """

        # Read queries
        queries_json = json.load(open(args.dataset + "cran_queries.json", 'r'))[:]
        query_ids, queries = [item["query number"] for item in queries_json], \
                                [item["query"] for item in queries_json]
        # Process queries 
        processedQueries = self.preprocessQueries(queries)

        # Read documents
        docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
        doc_ids, docs = [item["id"] for item in docs_json], \
                                [item["body"] for item in docs_json]
        # Process documents
        processedDocs = self.preprocessDocs(docs)
        
        qrels = json.load(open(args.dataset + "cran_qrels.json", 'r'))[:]
        

          
        # Build document index
        scores_bow = []
        for num_comp in range(20,1400,20):
            self.informationRetriever.buildIndex(processedDocs, doc_ids, num_comp, 'bow')
            doc_IDs_ordered = self.informationRetriever.rank(processedQueries)
            nDCG = self.evaluator.meanNDCG(
                doc_IDs_ordered, query_ids, qrels, 10)
            print("num_comp = ", num_comp, " score = ",nDCG)
            scores_bow.append(nDCG)

        # Plot the metrics and save plot 

        plt.plot(range(20,1400,20), scores_bow, label="nDCG")
        #plt.legend()
        plt.title("Evaluation Metrics (BOW Model)- Cranfield Dataset")
        plt.xlabel("num_components")
        # plt.show()
        plt.savefig(args.out_folder + "bowSelect.png")  


    def handleCustomQuery(self):
        """
        Take a custom query as input and return top five relevant documents
        """

        #Get query
        print("Enter query below")
        query = input()
        # Process documents
        processedQuery = self.preprocessQueries([query])[0]

        # Read documents
        docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
        doc_ids, docs = [item["id"] for item in docs_json], \
                            [item["body"] for item in docs_json]
        # Process documents
        processedDocs = self.preprocessDocs(docs)

        # Build document index
        self.informationRetriever.buildIndex(processedDocs, doc_ids)
        # Rank the documents for the query
        doc_IDs_ordered = self.informationRetriever.rank([processedQuery])[0]

        # Print the IDs of first five documents
        print("\nTop five document IDs : ")
        for id_ in doc_IDs_ordered[:5]:
            print(id_)



if __name__ == "__main__":

    # Create an argument parser
    parser = argparse.ArgumentParser(description='main.py')

    # Tunable parameters as external arguments
    parser.add_argument('-dataset', default = "cranfield/", 
                        help = "Path to the dataset folder")
    parser.add_argument('-out_folder', default = "output/", 
                        help = "Path to output folder")
    parser.add_argument('-segmenter', default = "punkt",
                        help = "Sentence Segmenter Type [naive|punkt]")
    parser.add_argument('-tokenizer',  default = "ptb",
                        help = "Tokenizer Type [naive|ptb]")
    parser.add_argument('-custom', action = "store_true", 
                        help = "Take custom query as input")
    
    # Parse the input arguments
    args = parser.parse_args()

    # Create an instance of the Search Engine
    searchEngine = SearchEngine(args)

    if args.custom:
        searchEngine.handleCustomQuery()
    else:
        searchEngine.evaluateDataset()
