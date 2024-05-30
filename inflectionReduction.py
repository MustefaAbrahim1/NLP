from util import *  # Import necessary utilities
from nltk.stem import WordNetLemmatizer  # Import WordNetLemmatizer from NLTK

class InflectionReduction:

    def reduce(self, text):
        """
        Stemming/Lemmatization

        Parameters
        ----------
        text : list
            A list of lists where each sub-list is a sequence of tokens
            representing a sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of
            stemmed/lemmatized tokens representing a sentence
        """

        lemmatizer = WordNetLemmatizer()  # Initialize WordNetLemmatizer
        wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}  # Map POS tags to WordNet tags

        def lemmatize_words(token):
            pos_tagged_text = nltk.pos_tag(token.split())  # Perform POS tagging on the tokenized words
            return [lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text][0]  # Lemmatize each word based on its POS tag

        reduced_text = []  # Initialize the list to store reduced text
        for sentence in text:  # Iterate through each sentence in the input text
            reduced_sentence = []  # Initialize the list to store reduced tokens for the current sentence
            for token in sentence:  # Iterate through each token in the current sentence
                reduced_sentence.append(lemmatize_words(token))  # Lemmatize the token and add it to the reduced sentence
            reduced_text.append(reduced_sentence)  # Add the reduced sentence to the reduced text
        return reduced_text  # Return the reduced text
