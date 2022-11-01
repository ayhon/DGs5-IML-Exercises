import numpy as np
from matplotlib import pyplot as plt
'''
Helper file includes supporting functions for notebook exercises
'''


'''
******************
****** K-NN ******
******************
'''

class KNNHelper:
	
	def __init__(self):

		# Weights, heights of individuals with known body category
		self.features_annotated_path = "data/hbody_feats_annotated.npy"     
		# Body categories of those individuals
		self.labels_annotated_path = "data/hbody_labels_annotated.npy"    
		# classes  
		self.class_names = np.array(['Underweight', 'Normal weight', 'Overweight'])	
		self.best_label=self.nn_indices=self.w=None
		self.num_classes = len(self.class_names)

		#loading data
		self.features_annotated, self.labels_annotated = self.load_data()
		        

	# load the fancy data
	def load_data(self):

		features_annotated = np.load(self.features_annotated_path)
		labels_annotated = np.load(self.labels_annotated_path)
		return features_annotated, labels_annotated

	# distance measure
	def manhattan_dist(self, example, training_examples):
	    return np.abs(training_examples - example).sum(axis=1)

	# Finds the indices of the k shortest distances from a list of distances
	def find_k_nearest_neighbors(self, k, distances):
	    indices = np.argsort(distances)[:k]
	    return indices

	# accuracy between a predicted and the actual labels.
	def accuracy(self, predicted, target):
	    return np.mean(predicted == target)

	# run K-NN for single example
	def weighted_kNN_one_example(self, unlabeled_example, training_features, training_labels, k, predict_label_func, verbose):

 		# Compute distances
	    distances = self.manhattan_dist(unlabeled_example, training_features) 
	    # Find neighbors
	    nn_indices = self.find_k_nearest_neighbors(k, distances)  
	    # Get neighbors' labels             
	    neighbor_labels = training_labels[nn_indices]                     
	    # Get neighbor's distances
	    nn_distances = distances[nn_indices]
	    #predict the label
	    best_label,w = predict_label_func(neighbor_labels, nn_distances, self.num_classes)   
	    if verbose:
	        for j in range(len(neighbor_labels)):
	            print(f"{j}th nearest neighbour with label {neighbor_labels[j]} contributes weight of {w[j]}")
	        print(f'Predicted label: {best_label}')
	    return best_label, nn_indices, w

	# find the label for a single example
	def find_label_with_weighted_KNN(self, predict_func,query_sample,verbose=False):
		
		total_samples = self.features_annotated.shape[0]
		all_indices = np.arange(total_samples)
		train_indices = np.delete(all_indices,query_sample)

		# Get the features corresponding to known and unknown points
		self.train_labels = self.labels_annotated[train_indices]
		self.train_feats = self.features_annotated[train_indices,:]
		self.query_feat =  self.features_annotated[query_sample,:]

		# choose number of neighbors
		k = 10

		# find the best label
		self.best_label,self.nn_indices,self.w = self.weighted_kNN_one_example(self.query_feat, self.train_feats, self.train_labels, k, predict_func,verbose)

# 		if visualize:
#  			self.plot_k_nearest_neighbors(query_feat, train_feats, train_labels, self.nn_indices)


	def plot_k_nearest_neighbors(self):
		query_feat = self.query_feat
		train_feats = self.train_feats 
		train_labels = self.train_labels
		nn_indices = self.nn_indices
		colors = np.array([[0.85, 0.85, 0], [0, 0.5, 0], [0.25, 0.25, 1]])
		plt.figure(figsize=(9,9))
		plt.title(f"A randomly chosen unlabeled example from the validation set\nand its {10}-nearest neighbors")
		for i, class_name in enumerate(self.class_names):
		    plt.scatter(*train_feats[train_labels==i].T,
		                c=colors[i, None], alpha=0.25, s=15, lw=0, label=class_name)
		for i, class_name in enumerate(self.class_names):
		    class_indices = nn_indices[train_labels[nn_indices] == i]
		    if len(class_indices) > 0:
		        plt.scatter(*train_feats[class_indices].T,
		                    c=colors[i, None], alpha=1, s=25, lw=0, label='Neighbor')

		ax = plt.scatter(*query_feat, marker='*', c='brown', alpha=0.5, s=50, label='unlabeled example')
		plt.xlabel("Weight (normalized)")
		plt.ylabel("Height (normalized)")
		plt.gca().set_aspect('equal')
		plt.legend();
   
	def predict_label(self, neighbor_labels):
		return np.argmax(np.bincount(neighbor_labels))


	def kNN_one_example(self, unlabeled_example, training_features, training_labels, k):
	    # WRITE YOUR CODE HERE
	    distances = self.manhattan_dist(unlabeled_example, training_features)  # Compute distances
	    nn_indices = self.find_k_nearest_neighbors(k, distances)               # Find neighbors
	    neighbor_labels = training_labels[nn_indices]                     # Get neighbors' labels
	    best_label = self.predict_label(neighbor_labels)                       # Pick the most common
	    
	    return best_label


	def kNN(self, unlabeled, training_features, training_labels, k):
		return np.apply_along_axis(self.kNN_one_example, 1, unlabeled, training_features, training_labels, k)


class KMEANSHelper:
    def __init__(self):
        self.data_path = 'to_add'

    def compute_centers(data, labels, K):
        """
        compute the center of each cluster
    
        input: 
    
        data:input data, shape is (N,F) where N is the number of samples, F is number of features
        labels: the assigned label of each data sample, shape is (N,1)
        K: the number of clusters
    
        output:
        centers: the new centers of each cluster, shape is(K,F) where K is the number of clusters and F is the number of features 
    
        """
        centers = np.zeros((K, data.shape[1]))
        for k in range(K):
            centers[k, :] = np.mean(data[labels == k, :], axis=0)
        return centers

