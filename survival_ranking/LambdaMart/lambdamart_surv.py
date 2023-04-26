import numpy as np
import math
import random
import copy
from sklearn.tree import DecisionTreeRegressor
from multiprocessing import Pool
from RegressionTree import RegressionTree
import pandas as pd
import pickle
from sksurv.metrics import concordance_index_censored
from tqdm import tqdm
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from numba import njit, prange

import pyximport; pyximport.install()
from util import compute_lambda

# New training including censoring
# original: [relevance score, query index, feature vector]
# new: 		[time, event  				 , feature vector]

# @njit(parallel=True)
# def compute_delta_one_step_smart(y_pred, y_true, event):
#     n = len(y_true)
#     delta = 0

#     for i in prange(n):
#         if (event[i]):
#             for j in range(n):

#                 if (i == 0) or (j == n-1):

#                     if(y_true[i] < y_true[j]):

#                         y_pred_sw_i = y_pred[i]
#                         y_pred_sw_j = y_pred[j]

#                         if i == 0:
#                             y_pred_sw_i = y_pred[-1]
#                         if j == n-1:
#                             y_pred_sw_j = y_pred[0]

#                         if (y_pred[i] < y_pred[j]) and (y_pred_sw_i > y_pred_sw_j):
#                             delta += 1
                        
#                         elif (y_pred[i] > y_pred[j]) and (y_pred_sw_i < y_pred_sw_j):
#                             delta += -1
  
#     return abs(delta)


# def compute_lambda(true_times, 
#                    true_events, 
#                    predicted_scores, 
#                    good_ij_pairs, all_good_ij_pairs):

#     num_evs = len(true_times) 
#     lambdas = np.zeros(num_evs, dtype=np.float64)
#     w = np.zeros(num_evs, dtype=np.float64)

#     for i,j in good_ij_pairs:
  
#         z_ndcg = compute_delta_one_step_smart(predicted_scores[i:j+1], true_times[i:j+1], true_events[i:j+1])/all_good_ij_pairs

#         rho = 1 / (1 + np.exp(predicted_scores[i] - predicted_scores[j]))
#         rho_complement = 1.0 - rho
#         lambda_val = z_ndcg * rho
#         lambdas[i] += lambda_val
#         lambdas[j] -= lambda_val

#         w_val = rho * rho_complement * z_ndcg
#         w[i] += w_val
#         w[j] += w_val

#     return lambdas, w

def get_labels_rsf(t, e):
    
    y = np.zeros(len(t), dtype = {'names': ('e', 't'),
                                            'formats': ('bool', 'i4')})

    y['e'] = e > 0
    y['t'] = t

    return y

def get_pairs(scores, events):
 
	pairs = []
	for i in range(len(scores)):
		for j in range(len(scores)):
			if (scores[i] < scores[j]) and (events[i]):
				pairs.append((i,j))
    
	return pairs


class LambdaMART:

	def __init__(self, training_data=None, number_of_trees=5, learning_rate=0.1, tree_type='sklearn'):
		"""
		This is the constructor for the LambdaMART object.
		Parameters
		----------
		training_data : list of int
			Contain a list of numbers
		number_of_trees : int (default: 5)
			Number of trees LambdaMART goes through
		learning_rate : float (default: 0.1)
			Rate at which we update our prediction with each tree
		tree_type : string (default: "sklearn")
			Either "sklearn" for using Sklearn implementation of the tree of "original" 
			for using our implementation
		"""

		if tree_type != 'sklearn' and tree_type != 'original':
			raise ValueError('The "tree_type" must be "sklearn" or "original"')
		self.training_data = training_data
		self.number_of_trees = number_of_trees
		self.learning_rate = learning_rate
		self.trees = []
		self.tree_type = tree_type
    
	def fit(self):
		self.training_data = self.training_data[self.training_data[:, 0].argsort()]
  
		# y_train = get_labels_rsf(self.training_data[:, 0], self.training_data[:, 1])
		# model_cox = CoxnetSurvivalAnalysis().fit(self.training_data[:, 2:], y_train)
  
		# predicted_scores = -model_cox.predict(self.training_data[:, 2:])
		# predicted_scores = predicted_scores/max(predicted_scores)
		predicted_scores = np.asarray(random.choices(np.arange(0, len(self.training_data)), k=len(self.training_data))).astype(float) # np.zeros(len(self.training_data))
		predicted_scores = (predicted_scores-min(predicted_scores))/max(predicted_scores)

		true_scores = self.training_data[:, 0]
		true_events = self.training_data[:, 1]
		# all_good_ij_pairs = len(get_pairs(true_scores, true_events))


		list_idxs = 			[np.random.choice(list(range(len(self.training_data))), size=int(len(self.training_data)), replace=True) \
      								for _ in range(self.number_of_trees)]
		list_idxs = 			[np.sort(idxs) for idxs in list_idxs]
  
		list_good_ij_pairs = 	[get_pairs(true_scores[idxs], true_events[idxs]) for idxs in list_idxs]
  
		all_good_ij_pairs = 	[len(pairs) for pairs in list_good_ij_pairs]
		all_good_ij_pairs = 	np.sum(np.asarray(all_good_ij_pairs))
  
		list_true_scores = 		[true_scores[idxs] for idxs in list_idxs]
		list_true_events = 		[true_events[idxs] for idxs in list_idxs]

		for k in range(self.number_of_trees):

			lambdas = np.zeros(len(list_true_scores[k]))
 
			lambdas, w = compute_lambda(list_true_scores[k], \
       			list_true_events[k].astype('int32'), predicted_scores[list_idxs[k]], list_good_ij_pairs[k], all_good_ij_pairs)

			tree = DecisionTreeRegressor(min_samples_split=6, min_samples_leaf=3)
			tree.fit(self.training_data[:,2:][list_idxs[k]], lambdas, sample_weight=w)
			self.trees.append(tree)
			prediction = tree.predict(self.training_data[:,2:])
			predicted_scores += prediction * self.learning_rate


	def predict(self, data):
		"""
		Predicts the scores for the test dataset.
		Parameters
		----------
		data : Numpy array of documents
			Numpy array of documents with each document's format is [query index, feature vector]
		
		Returns
		-------
		predicted_scores : Numpy array of scores
			This contains an array or the predicted scores for the documents.
		"""
		data = np.array(data)
		predicted_scores = np.zeros(len(data))
		results = np.zeros(len(data))
		for tree in self.trees:
			results += self.learning_rate * tree.predict(data[:, 2:])
		predicted_scores = results
		return predicted_scores

	def validate(self, data):
		"""
		Predicts the scores for the test dataset and calculates the NDCG value.
		Parameters
		----------
		data : Numpy array of documents
			Numpy array of documents with each document's format is [relevance score, query index, feature vector]
		k : int
			this is used to compute the NDCG@k
		
		Returns
		-------
		average_ndcg : float
			This is the average NDCG value of all the queries
		predicted_scores : Numpy array of scores
			This contains an array or the predicted scores for the documents.
		"""
  
		data = np.array(data)
		results = np.zeros((len(data)))
		for tree in self.trees:
			results += self.learning_rate * tree.predict(data[:, 2:])

		cidx1 = concordance_index_censored(data[:, 1]>0, data[:, 0], results)[0]
  
		return cidx1

	# def save(self, fname):
	# 	"""
	# 	Saves the model into a ".lmart" file with the name given as a parameter.
	# 	Parameters
	# 	----------
	# 	fname : string
	# 		Filename of the file you want to save
		
	# 	"""
	# 	pickle.dump(self, open('%s.lmart' % (fname), "wb"), protocol=2)

	# def load(self, fname):
	# 	"""
	# 	Loads the model from the ".lmart" file given as a parameter.
	# 	Parameters
	# 	----------
	# 	fname : string
	# 		Filename of the file you want to load
		
	# 	"""
	# 	model = pickle.load(open(fname , "rb"))
	# 	self.training_data = model.training_data
	# 	self.number_of_trees = model.number_of_trees
	# 	self.tree_type = model.tree_type
	# 	self.learning_rate = model.learning_rate
	# 	self.trees = model.trees