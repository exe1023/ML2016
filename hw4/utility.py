from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import *
import csv
import sys
import string
import numpy as np

def getCorpus(dire):
	path = dire + '/title_StackOverflow.txt'
	corpus = []
	cachedstopwords = stopwords.words('english')
	wnl = WordNetLemmatizer()
	with open(path, 'r',encoding='utf-8') as f:
		for line in f:
			line = line.lower()
			line = ''.join([cha if cha not in string.punctuation else ' ' for cha in line])
			corpus.append(' '.join([word for word in line.split() if word not in cachedstopwords]))
	return corpus


def TFIDF(corpus, hash=False):
	if hash == True:
		hasher = HashingVectorizer()
		vectorizer = make_pipeline(hasher, TfidfTransformer())
	else:
		vectorizer = TfidfVectorizer(max_features=30000,ngram_range=(1,1),
			stop_words='english', use_idf=True)
	tfidf = vectorizer.fit_transform(corpus)

	tfidffeature = vectorizer.get_feature_names()
	print('tfidf feature num =', len(tfidffeature) )
	vocabulary = vectorizer.vocabulary_
	idf = vectorizer.idf_

	return tfidf,tfidffeature

def LSA(x, dim=20):
	print("LSA with dim = %d" %dim)
	svd = TruncatedSVD(dim)
	normalizer = Normalizer(copy=False)
	lsa = make_pipeline(svd, normalizer)

	return lsa.fit_transform(x)

def getCheck(dire):
	path = dire + '/check_index.csv'
	check = []
	with open(path, 'r') as f:
		f.readline() # ignore the first line
		for line in csv.reader(f):
			sample = list(map(int,line[1:]))
			check.append(sample)

	return check

def AvgVector(words, model, index2word_set, tfidf, tfidffeature, count):
	featureVec = np.zeros((500,))
	nwords = 0

	#print("--------")
	for word in words.split():
		if word in index2word_set:
			#print(word)
			#try:
			#	index = tfidffeature.index(word)
			#except:
			#	continue
			#featureVec += tfidf[count,index] * model[word]
			featureVec += model[word]
			nwords += 1
	#print("--------")
	if nwords == 0:
		nwords = 1

	featureVec = featureVec / nwords
	return featureVec

def c2v(model, corpus, tfidf, tfidffeature):
	feature = []
	index2word_set = set(model.index2word)
	#print(index2word_set)
	count = 0
	for sentence in corpus:
		feature.append(AvgVector(sentence, model, index2word_set, tfidf, tfidffeature, count))
		count += 1

	return feature

def preprocessDocs(dire):
	path = dire + '/docs.txt'
	cachedstopwords = stopwords.words('english')
	with open(path, 'r',encoding='utf-8') as f:
		with open(dire + 'docsD.txt', 'w',encoding='utf-8') as out:
			sys.stdout = out
			for line in f:
				line = line.lower()
				line = ''.join([cha if cha not in string.punctuation else ' ' for cha in line])
				line = ' '.join([word for word in line.split() if word not in cachedstopwords])
				if len(line.split()) > 3:
					print(line)
			sys.stdout = sys.__stdout__

def w2v(dire):
	print("start training w2v model")
	
	docs = preprocessDocs(dire)
	model = word2vec.Word2Vec(word2vec.LineSentence(dire+'docsD.txt'), size=500, iter=25, sg=1,hs=1,window=20,
		sample=1e-5)
	model.save("w2vmodel")
	
	print("end traning")
	return model

def outputCheck(dire, result, file):
	checklist = getCheck(dire)
	with open(file, 'w') as f:
		sys.stdout = f
		print("ID,Ans")
		checkid = 0
		for line in checklist:
			if result[line[0]] == result[line[1]]:
				print("%d,1"%checkid)
			else:
				print("%d,0"%checkid)
			checkid += 1
		sys.stdout = sys.__stdout__


