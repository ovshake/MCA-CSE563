
from numpy import dot
from numpy.linalg import norm
import numpy as np 

def cos_sim(a,b):
	return dot(a, b)/(norm(a)*norm(b))


def relevance_feedback(vec_docs, vec_queries, sim, n=10):
	"""
	relevance feedback
	Parameters
		----------
		vec_docs: sparse array,
			tfidf vectors for documents. Each row corresponds to a document.
		vec_queries: sparse array,
			tfidf vectors for queries. Each row corresponds to a document.
		sim: numpy array,
			matrix of similarities scores between documents (rows) and queries (columns)
		n: integer
			number of documents to assume relevant/non relevant

	Returns
	-------
	rf_sim : numpy array
		matrix of similarities scores between documents (rows) and updated queries (columns)
	"""
	# rf_sim = sim # change
	# alpha = 0.7
	# beta = 0.3 best 
	num_docs = vec_docs.shape[0] 
	num_queries = vec_queries.shape[0]
	alpha = 0.3
	beta = 0.7
	num_iters = 3
	for q_idx, q in enumerate(vec_queries):
		# print(q.shape)
		for iter in range(num_iters):
			sim_q = sim[:,q_idx] 
			sim_q = [(s, doc_id) for doc_id,s in enumerate(sim_q)]
			sim_q.sort(reverse=True) 
			rel_doc = sim_q[:n] 
			nonrel_doc = sim_q[-n:] 
			for doc in rel_doc:
				q += alpha * vec_docs[doc[1]]

			for doc in nonrel_doc:
				q -= beta * vec_docs[doc[1]]


			for doc_id in range(num_docs):
				cur_sim = cos_sim(q.todense(), vec_docs[doc_id].todense().T) 
				# print('cur sim ', cur_sim[0,0])
				sim[doc_id][q_idx] = cur_sim[0,0]
		
	rf_sim = sim 
	return rf_sim


def get_similar_words(word_sim_mat, K, word):
	total_words = word_sim_mat.shape[1]
	sim_tuple_array = []
	for i in range(total_words):
		if i == word:
			continue
		sim_tuple_array.append((word_sim_mat[word, i], i)) 

	sim_tuple_array.sort(reverse=True) 
	sim_tuple_array = sim_tuple_array[:K] 
	return [i[1] for i in sim_tuple_array] 

def arg_max(vec, P):
	sim_tuple_array = []
	total_words = vec.shape[1] 
	for i in range(total_words):
		sim_tuple_array.append((vec[0,i], i)) 
	
	sim_tuple_array.sort(reverse=True) 
	sim_tuple_array = sim_tuple_array[:P]

	return [i[1] for i in sim_tuple_array]



def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, n=10, k = 1):
	"""
	relevance feedback with expanded queries
	Parameters
		----------
		vec_docs: sparse array,
			tfidf vectors for documents. Each row corresponds to a document.
		vec_queries: sparse array,
			tfidf vectors for queries. Each row corresponds to a document.
		sim: numpy array,
			matrix of similarities scores between documents (rows) and queries (columns)
		tfidf_model: TfidfVectorizer,
			tf_idf pretrained model
		n: integer
			number of documents to assume relevant/non relevant

	Returns
	-------
	rf_sim : numpy array
		matrix of similarities scores between documents (rows) and updated queries (columns)
	"""
	term_sim_vec = vec_docs.T.dot(vec_docs)
	# print(term_sim_vec.shape)


	# rf_sim = sim  # change
	# rf_sim = sim # change
	num_docs = vec_docs.shape[0] 
	num_queries = vec_queries.shape[0]
	alpha = 0.5
	beta = 0.5
	num_iters = 3
	for q_idx, q in enumerate(vec_queries):
		for iter in range(num_iters):
			common_words = arg_max(q, 1) #q.argmax()
			for common_word in common_words:
				cur_wt = q[0,common_word] 
				sim_words = get_similar_words(term_sim_vec, k, common_word) 
				for idx in sim_words:
					q[0,idx] = cur_wt

			sim_q = sim[:,q_idx] 
			sim_q = [(s, doc_id) for doc_id,s in enumerate(sim_q)]
			sim_q.sort(reverse=True) 
			rel_doc = sim_q[:n] 
			nonrel_doc = sim_q[-n:] 
			for doc in rel_doc:
				q += alpha * vec_docs[doc[1]]

			for doc in nonrel_doc:
				q -= beta * vec_docs[doc[1]]


			for doc_id in range(num_docs):
				cur_sim = cos_sim(q.todense(), vec_docs[doc_id].todense().T) 
				# print('cur sim ', cur_sim[0,0])
				sim[doc_id][q_idx] = cur_sim[0,0]
		
	rf_sim = sim
	return rf_sim



# Baseline Retrieval
# MAP: 0.5183859040856561

# Retrieval with Relevance Feedback
# MAP: 0.6087861794264364

# Retrieval with Relevance Feedback and query expansion
# MAP: 0.6202550578718745