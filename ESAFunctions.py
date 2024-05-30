from mediawiki import MediaWiki
import numpy as np

def wikipediaDocs(docs_title):
    """
    Retrieve summaries of Wikipedia pages related to the given document titles.

    Args:
    docs_title (list): List of document titles.

    Returns:
    list: List of Wikipedia page summaries.
    """
    wikipedia = MediaWiki()
    wikipedia_docs = []
    
    # Iterate over each document title
    for title in docs_title:
        try:
            # Search for the document title on Wikipedia
            page_list = wikipedia.search(title)
        except:
            pass
        
        # Retrieve summaries for up to 5 search results
        for i in range(min(len(page_list), 5)):
            page_name = page_list[i]
            try:
                # Get the Wikipedia page object
                p = wikipedia.page(page_name)
            except:
                pass
            
            # Append the summary of the Wikipedia page to the list
            wikipedia_docs.append(p.summary)
    
    return wikipedia_docs

def get_ESA_vector(doc_list, tf_idf, DocsWiki, wordIndex, Docs, wordIndexWiki):
    """
    Compute the Enhanced Semantic Analysis (ESA) vector for the given document list.

    Args:
    doc_list (list): List of tokens representing the document.
    tf_idf (float): TF-IDF values associated with terms in the document.
    DocsWiki (numpy.ndarray): TF-IDF matrix for Wikipedia documents.
    wordIndex (dict): Dictionary mapping tokens to their indices in the TF-IDF matrix for the document.
    Docs (numpy.ndarray): TF-IDF matrix for the document.
    wordIndexWiki (dict): Dictionary mapping tokens to their indices in the TF-IDF matrix for Wikipedia documents.

    Returns:
    numpy.ndarray: ESA vector for the document.
    """
    esa_vec_final = np.zeros(DocsWiki.shape[0])
    
    # Iterate over tokens in the document list
    for tokens in doc_list:
        tf_idf_value = 0.0
        esa_vec = np.zeros(DocsWiki.shape[0])
        
        # Get TF-IDF value for the token in the document
        try:
            row_carn, col_carn = wordIndex[tokens]
            tf_idf_value = Docs[row_carn, col_carn]
        except:
            row_carn, col_carn = 0, 0
        
        # Retrieve ESA vector for the token from Wikipedia TF-IDF matrix
        try:
            row_wiki, col_wiki = wordIndexWiki[tokens]
            esa_vec = tf_idf_value * DocsWiki[:, col_wiki]
        except:
            row_wiki, col_wiki = 0, 0
        
        # Aggregate ESA vectors
        esa_vec_final = esa_vec_final + esa_vec
    
    return esa_vec_final
