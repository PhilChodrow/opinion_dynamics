import numpy as np
from scipy.sparse import lil_matrix, dok_matrix, csr_matrix
import pandas as pd

class sparse_coev:

	def __init__(self, N, c, k):

		q = 1.0 * c / N

		A    = csr_matrix((N, N))
		B    = np.random.choice([0, 1], size=(N,N), p=[1-q, q])
		B    = (B + B.T > 0) * 1
		A[:] = B

		v = np.random.choice(np.arange(k), size = N)

		self.v = v
		self.A = A
		self.P = csr_matrix(A.shape)

		self.n = len(v)
		self.m = A.sum()

		self.update_tension(edge_list_add = None)

	def update_tension(self, edge_list_add, edge_list_subtract = None):
		'''
		edge_list a 2-tuple of vectors giving the indices
		'''

		if edge_list_add is None:
			edge_list_add = self.A.nonzero()

		edge_list_add_t = (edge_list_add[1], edge_list_add[0])
		for lists in [edge_list_add, edge_list_add_t]:
			self.P[lists] = [abs(self.v[i] - self.v[j]) for (i,j) in zip(*lists)]

		if edge_list_subtract is not None:
			edge_list_subtract_t = (edge_list_subtract[1], edge_list_subtract[0])
			for lists in [edge_list_subtract, edge_list_subtract_t]:
				self.P[lists] = [0 for (i,j) in zip(*lists)]
		
	def sample_edge(self, weighted = False):
		'''
		weight controls whether the sampling is proportional to the tension of the edge
		'''

		edge_list = self.P.nonzero() # only need ones with nonzero tension
		k = len(edge_list[0])

		if weighted:	
			a = self.P[edge_list].A1
			a = a / a.sum()
		else:
			a = np.ones(k) / k
		
		i = np.random.choice(range(k), p = a)
		return edge_list[0][i], edge_list[1][i]

	def rewire_edge(self, i, j):

		if self.A[i,j] < .5:
			# print(self.A[i,j])
			# print "No edge between " + str(i) + " and " + str(j) + "."
			return None
		else:
			# print "rewiring"
			end = np.random.choice([i,j])               # pick one of i and j to be the base
			other_end = np.random.choice(range(self.n))      # pick a random node
			while end == other_end:  					# in case it's end, just pick another one
				other_end = np.random.choice(range(self.n))

		self.A[i,j] = 0            # delete existing
		self.A[j,i] = 0            # delete existing
	
		self.A[end, other_end] = 1 # add new
		self.A[other_end, end] = 1 # add new

		self.update_tension(edge_list_add = (np.array([end]), np.array([other_end])),
		                    edge_list_subtract = (np.array([i]), np.array([j])))

	def argue(self, i,j):
		self.v[i] = self.v[i] + np.sign(self.v[j] - self.v[i])

	def update_step(self, alpha, weighted = False):
		
		e = self.sample_edge(weighted)

		if(np.random.rand() < alpha): # rewire
			self.rewire_edge(*e)
			
		else:
			ind = np.random.choice([0,1])
			i = e[ind]
			j = e[1-ind]

			self.argue(i,j)

			ego = self.A[i].nonzero()[1]
			edge_list = (np.repeat(i, len(ego)), ego)
			
			self.update_tension(edge_list_add = edge_list)

	def get_v(self):
		return self.v

	def get_P(self):
		return self.P

	def mean_tension(self):
		return self.P[self.P.nonzero()].sum() / self.m

	def variance(self):
		return np.var(self.v)

	def mean_v(self):
		return self.v.mean()

	def percent_tension(self):
		return 1.0 * (self.P > 0).sum() / self.m

	def mean_square_tension(self):
		return 1.0 * np.power(self.P, 2).sum() / self.m

	def dynamics(self, nsteps = None, alpha = .5,  verbose = False, interval = None, notify_end = True):
		
		tension_list = list()
		variance_list = list()
		mean_list = list()
		mean_tension_list = list()
		mean_square_tension_list = list()

		i = 0
		done = False
		
		while not done:
			i += 1
			self.update_step(alpha = alpha)
			mean_tension = self.mean_tension()
			variance = self.variance()

			if nsteps is not None:		
				done = i >= nsteps
			
			done = done or mean_tension == 0

			if verbose:
				if (i % interval == 0):
				    print i, round(mean_tension, 2), round(variance, 2)				 

			# tension_list.append(self.percent_tension())
			# variance_list.append(self.variance())
			# mean_list.append(self.mean_v())
			# mean_tension_list.append(mean_tension)
			# mean_square_tension_list.append(self.mean_square_tension())

		
		d = {'tension'      : np.array(tension_list),
			 'variance'     : np.array(variance_list),
			 'mean_opinion' : np.array(mean_list),
			 # 'mean_square_tension' : np.array(mean_square_tension_list),
			 'mean_tension' : np.array(mean_tension_list)}

		df = pd.DataFrame(d)
		df['t'] = np.arange(len(df))
		
		if notify_end:
			print 'Done in ' + str(i) + ' steps.'
		return df

def run_dynamics(c, k, N, alpha, verbose = False, nsteps = None):
    coev = sparse_coev(N, c, k)
    df = coev.dynamics(alpha = alpha, 
                       verbose = verbose, 
                       interval = 1000, 
                       nsteps = nsteps)

    param_cols = {'c' : c, 'k' : k, 'N' : N, 'alpha' : alpha}
    for col_name in param_cols:
        df[col_name] = np.repeat(param_cols[col_name], len(df))
    
    return df