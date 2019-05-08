import numpy as np
import tensorflow as tf

class Code:
	def __init__(self):
		self.num_edges = 0
		# self.n = n
		# self.k = k

def load_code(H_filename, G_filename):
	# parity-check matrix; Tanner graph parameters
	with open(H_filename) as f:
		# get n and m (n-k) from first line
		n,m = [int(s) for s in f.readline().split(' ')]
		k = n-m

		var_degrees = np.zeros(n).astype(np.int) # degree of each variable node
		chk_degrees = np.zeros(m).astype(np.int) # degree of each check node

		# initialize H
		H = np.zeros([m,n]).astype(np.int)
		max_var_degree, max_chk_degree = [int(s) for s in f.readline().split(' ')]
		f.readline() # ignore two lines
		f.readline()

		# create H, sparse version of H, and edge index matrices
		# (edge index matrices used to calculate source and destination nodes during belief propagation)
		var_edges = [[] for _ in range(0,n)]
		for i in range(0,n):
			row_string = f.readline().split(' ')
			var_edges[i] = [(int(s)-1) for s in row_string[:-1]]
			var_degrees[i] = len(var_edges[i])
			H[var_edges[i], i] = 1

		chk_edges = [[] for _ in range(0,m)]
		for i in range(0,m):
			row_string = f.readline().split(' ')
			chk_edges[i] = [(int(s)-1) for s in row_string[:-1]]
			chk_degrees[i] = len(chk_edges[i])

		d = [[] for _ in range(0,n)]
		edge = 0
		for i in range(0,n):
			for j in range(0,var_degrees[i]):
				d[i].append(edge)
				edge += 1

		u = [[] for _ in range(0,m)]
		edge = 0
		for i in range(0,m):
			for j in range(0,chk_degrees[i]):
				v = chk_edges[i][j]
				for e in range(0,var_degrees[v]):
					if (i == var_edges[v][e]):
						u[i].append(d[v][e])

		num_edges = H.sum()

	if G_filename == "":
		G = []
	else:
		if "BCH" in H_filename: # dear God please fix this
			G = np.loadtxt(G_filename).astype(np.int)
			G = G.transpose()
		else:
			P = np.loadtxt(G_filename,skiprows=2)
			G = np.vstack([P.transpose(), np.eye(k)]).astype(np.int)

	code = Code()
	code.H = H
	code.G = G
	code.var_degrees = var_degrees
	code.chk_degrees = chk_degrees
	code.num_edges = num_edges
	code.u = u
	code.d = d
	code.n = n
	code.m = m
	code.k = k
	return code


# compute the "soft syndrome"
def syndrome(soft_output, code):
	H = code.H
	G = code.G
	n = code.n
	m = code.m
	k = code.k
	soft_syndrome = []
	for c in range(0, m): # for each check node
		variable_nodes = []
		for v in range(0, n):
			if H[c,v] == 1: variable_nodes.append(v)
		temp = tf.gather(soft_output,variable_nodes)
		temp1 = tf.reduce_prod(tf.sign(temp),0)
		temp2 = tf.reduce_min(tf.abs(temp),0)
		soft_syndrome.append(temp1 * temp2)
	soft_syndrome = tf.stack(soft_syndrome)
	return soft_syndrome
