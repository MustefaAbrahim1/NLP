from util import *

class StopwordRemoval():

	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""

		STOPWORDS = set(stopwords.words('english'))

		def remove_stopwords(token, STOPWORDS):
			answer = []
			for i in token:
				if i not in STOPWORDS:
					answer += [i]
			return answer

		stopwordRemovedText = []
		for i in text:
			stopwordRemovedText += [remove_stopwords(i, STOPWORDS)]

		return stopwordRemovedText




	