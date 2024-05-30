# IRS:Information Retrieval System (NLP-Project-2024)

structure of our codes and files of IRS

-   **plot_result** This notebook contains all the techniques
    we have tried in this project like Vector Space Model with two
    versions (VSM-1 and VSM-2), Latent Semantic Analysis (LSA), Query
    Expansion(QE) + LSA, Clustering techniques to improve retrieval time
    and hypothesis testings.

-   **Query Processing**This is the additional technique we tried
    to help the user in finishing the query.

-   **sentenceSegmentation.py** This python file contains, different
    approaches to convert a document into list of sentences.

    -   ‘naive‘: Here splitting the document based on "pull stop"

    -   ’punkt’: Here we used punkt sentence tokenizer.

-   **stopwordRemoval.py** This file contains the process of removing
    stopwords like the, of, and,.. given a document. Here the list of
    stopwords are imported from nltk package.

-   **tokenization.py**This python file contains, different approaches
    to convert a sentence into list of words.

    -   ‘naive‘: Here splitting the sentence based on "pull stop",
        "comma" and other common delimiters

    -   ’pennTreeBank’: Here we used pennTreeBank tokenizer to tokenize
        the sentence.

-   **evaluation.py** This file contains all the evaluation metric
    functions like queryPrecision, queryRecall, meanPrecision and so on.

-   **inflectionReduction.py** For stemming or lemmatization. (i.e. to
    reduce a word to it’s root form).


-   **’run’:** plots the distribution of the recall, precision, ndcg,
    f-score.

-   **’run|comp’:** to compare the distributions of recall, precision,
     ndcg, f-score of any two models.

-   **tfidf.py** This file contains

    -   ’TF\_IDF’: gives tf\_idf representation of the documents.

-   **util.py** This file contains utility function.



-   **Folders**

    -   **cranfield:** This folder contains the training
        dataset(cran\_docs.json), queries to be
        evaluated(cran\_queries.json) and true order of the documents
        (cran\_qrels.json)
    - **Result**
         - Here we have report of IRS and Warm-up
    - **result**
          - some figures we used in our report and from the codes 

    - **output:** This folder contains the output files of the assignment
        2 i.e segmented documents into sentences, then sentences to
        tokens, then lemmatization, then stopword removal.
