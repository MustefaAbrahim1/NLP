from util import *
from nltk.tokenize.treebank import TreebankWordTokenizer
class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""
		tokenizedText = []  # Initialize an empty list to store tokenized sentences
		for sentence in text:  # Iterate through each sentence in the input text
			# Split the sentence into tokens using whitespace as the delimiter
			tokens = sentence.split()
			# Append the list of tokens to the tokenizedText list
			tokenizedText.append(tokens)
		return tokenizedText  # Return the list of tokenized sentences




	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""
		tokenizer = TreebankWordTokenizer()
		tokenizedText = [tokenizer.tokenize(sentence) for sentence in text]
		return tokenizedText