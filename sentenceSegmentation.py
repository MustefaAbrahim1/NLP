from util import *
import re
from nltk.tokenize import PunktSentenceTokenizer
class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach assuming sentences end with period, question mark, or exclamation mark.

		Parameters
		----------
		text : str
			A string containing multiple sentences

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""
		# Split the text on period, question mark, or exclamation mark
		segmented_text = re.split(r'[.!?]', text)
		# Remove empty strings and strip whitespace from each sentence
		segmented_text = [sentence.strip() for sentence in segmented_text if sentence.strip()]
		return segmented_text

		
	def punkt(self, text):

		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each strin is a single sentence
		"""

		tokenizer = PunktSentenceTokenizer() # initialize the tokenizer using panket
		segmentedText = tokenizer.tokenize(text)
		
		return segmentedText