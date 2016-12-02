from utility import *
from sklearn.cluster import KMeans,MiniBatchKMeans,AgglomerativeClustering
from gensim.models import *
import sys
import nltk

def main():
	corpus = getCorpus(sys.argv[1])
	tfidf,tfidffeaeture = TFIDF(corpus)
	lsafeature = LSA(tfidf)

	tokens = []
	for sentence in corpus[10:40]:
		pass
		#print(sentence)
		#tokens = nltk.word_tokenize(sentence)
		#tagged = nltk.pos_tag(tokens)
		#print(tagged)

	if len(sys.argv) >= 4:
		model = word2vec.Word2Vec.load(sys.argv[3])
	else:
		model = w2v(sys.argv[1])

	w2vfeature = c2v(model, corpus, tfidf, tfidffeaeture)
	w2vfeature = LSA(w2vfeature,dim=60)
	
	feature = w2vfeature
	cluster = KMeans(n_clusters=20, n_init=20, tol=0, verbose=0)
	#cluster = AgglomerativeClustering(n_clusters=20)
	cluster.fit(feature)

	print(cluster.inertia_)

	outputCheck(sys.argv[1], cluster.labels_, sys.argv[2])

if __name__ == '__main__':
	main()
