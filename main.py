# Belief propagation using TensorFlow
# Run as follows:
# python main.py 0 1 6 1 100 10000000000000000 5 hamming.alist hamming.gmat laskdjhf 0.5 100 FNOMS
import numpy as np
import tensorflow as tf
import sys
from tensorflow.python.framework import ops
from helper_functions import load_code, syndrome
import os

DEBUG = False
TRAINING = True
SUM_PRODUCT = False
MIN_SUM = not SUM_PRODUCT
ALL_ZEROS_CODEWORD_TRAINING = True 
ALL_ZEROS_CODEWORD_TESTING = False
NO_SIGMA_SCALING_TRAIN = False
NO_SIGMA_SCALING_TEST = False
np.set_printoptions(precision=3)

print("My PID: " + str(os.getpid()))

if SUM_PRODUCT:
	print("Using Sum-Product algorithm")
if MIN_SUM:
	print("Using Min-Sum algorithm")

if ALL_ZEROS_CODEWORD_TRAINING:
	print("Training using only the all-zeros codeword")
else:
	print("Training using random codewords (not the all-zeros codeword)")

if ALL_ZEROS_CODEWORD_TESTING:
	print("Testing using only the all-zeros codeword")
else:
	print("Testing using random codewords (not the all-zeros codeword)")

if NO_SIGMA_SCALING_TRAIN:
	print("Not scaling train input by 2/sigma")
else:
	print("Scaling train input by 2/sigma")

if NO_SIGMA_SCALING_TEST:
	print("Not scaling test input by 2/sigma")
else:
	print("Scaling test input by 2/sigma")

seed = int(sys.argv[1])
np.random.seed(seed)
snr_lo = float(sys.argv[2])
snr_hi = float(sys.argv[3])
snr_step = float(sys.argv[4])
min_frame_errors = int(sys.argv[5])
max_frames = float(sys.argv[6])
num_iterations = int(sys.argv[7])
H_filename = sys.argv[8]
G_filename = sys.argv[9]
output_filename = sys.argv[10]
L = float(sys.argv[11])
steps = int(sys.argv[12])
provided_decoder_type = sys.argv[13]

if ALL_ZEROS_CODEWORD_TESTING: G_filename = ""
code = load_code(H_filename, G_filename)

# code.H = np.array([[1, 1, 0, 1, 1, 0, 0],
#        [1, 0, 1, 1, 0, 1, 0],
#        [0, 1, 1, 1, 0, 0, 1]])

H = code.H
G = code.G
var_degrees = code.var_degrees
chk_degrees = code.chk_degrees
num_edges = code.num_edges
u = code.u
d = code.d
n = code.n
m = code.m
k = code.k

class Decoder:
	def __init__(self, decoder_type="RNOMS", random_seed=0, learning_rate = 0.001, relaxed = False):
		self.decoder_type = decoder_type
		self.random_seed = random_seed
		self.learning_rate = learning_rate
		self.relaxed = relaxed

# decoder parameters
batch_size = 120
tf_train_dataset = tf.placeholder(tf.float32, shape=(n,batch_size))
tf_train_labels = tf.placeholder(tf.float32, shape=(n,batch_size))#tf.placeholder(tf.float32, shape=(num_iterations,n,batch_size))

#### decoder functions ####

# compute messages from variable nodes to check nodes
def compute_vc(cv, iteration, soft_input):
	weighted_soft_input = soft_input
	
	edges = []
	for i in range(0, n):
		for j in range(0, var_degrees[i]):
			edges.append(i)
	reordered_soft_input = tf.gather(weighted_soft_input, edges)
	
	vc = []
	edge_order = []
	for i in range(0, n): # for each variable node v
		for j in range(0, var_degrees[i]):
			# edge = d[i][j]
			edge_order.append(d[i][j])
			extrinsic_edges = []
			for jj in range(0, var_degrees[i]):
				if jj != j: # extrinsic information only
					extrinsic_edges.append(d[i][jj])
			# if the list of edges is not empty, add them up
			if extrinsic_edges:
				temp = tf.gather(cv,extrinsic_edges)
				temp = tf.reduce_sum(temp,0)
			else:
				temp = tf.zeros([batch_size])
			if SUM_PRODUCT: temp = tf.cast(temp, tf.float32)#tf.cast(temp, tf.float64)
			vc.append(temp)
	
	vc = tf.stack(vc)
	new_order = np.zeros(num_edges).astype(np.int)
	new_order[edge_order] = np.array(range(0,num_edges)).astype(np.int)
	vc = tf.gather(vc,new_order)
	vc = vc + reordered_soft_input
	return vc

# compute messages from check nodes to variable nodes
def compute_cv(vc, iteration):
	cv_list = []
	prod_list = []
	min_list = []
	
	if SUM_PRODUCT:
		vc = tf.clip_by_value(vc, -10, 10)
		tanh_vc = tf.tanh(vc / 2.0)
	edge_order = []
	for i in range(0, m): # for each check node c
		for j in range(0, chk_degrees[i]):
			# edge = u[i][j]
			edge_order.append(u[i][j])
			extrinsic_edges = []
			for jj in range(0, chk_degrees[i]):
				if jj != j:
					extrinsic_edges.append(u[i][jj])
			if SUM_PRODUCT:
				temp = tf.gather(tanh_vc,extrinsic_edges)
				temp = tf.reduce_prod(temp,0)
				temp = tf.log((1+temp)/(1-temp))
				cv_list.append(temp)
			if MIN_SUM:
				temp = tf.gather(vc,extrinsic_edges)
				temp1 = tf.reduce_prod(tf.sign(temp),0)
				temp2 = tf.reduce_min(tf.abs(temp),0)
				prod_list.append(temp1)
				min_list.append(temp2)
	
	if SUM_PRODUCT:
		cv = tf.stack(cv_list)
	if MIN_SUM:
		prods = tf.stack(prod_list)
		mins = tf.stack(min_list)
		if decoder.decoder_type == "RNOMS":
			# offsets = tf.nn.softplus(decoder.B_cv)
			# mins = tf.nn.relu(mins - tf.tile(tf.reshape(offsets,[-1,1]),[1,batch_size]))
			mins = tf.nn.relu(mins - decoder.B_cv)
		elif decoder.decoder_type == "FNOMS":
			offsets = tf.nn.softplus(decoder.B_cv[iteration])
			mins = tf.nn.relu(mins - tf.tile(tf.reshape(offsets,[-1,1]),[1,batch_size]))
		cv = prods * mins
	
	new_order = np.zeros(num_edges).astype(np.int)
	new_order[edge_order] = np.array(range(0,num_edges)).astype(np.int)
	cv = tf.gather(cv,new_order)
	
	if decoder.decoder_type == "RNSPA" or decoder.decoder_type == "RNNMS":
		cv = cv * tf.tile(tf.reshape(decoder.W_cv,[-1,1]),[1,batch_size])
	elif decoder.decoder_type == "FNSPA" or decoder.decoder_type == "FNNMS":
		cv = cv * tf.tile(tf.reshape(decoder.W_cv[iteration],[-1,1]),[1,batch_size])
	return cv

# combine messages to get posterior LLRs
def marginalize(soft_input, iteration, cv):
	weighted_soft_input = soft_input

	soft_output = []
	for i in range(0,n):
		edges = []
		for e in range(0,var_degrees[i]):
			edges.append(d[i][e])

		temp = tf.gather(cv,edges)
		temp = tf.reduce_sum(temp,0)
		soft_output.append(temp)

	soft_output = tf.stack(soft_output)

	soft_output = weighted_soft_input + soft_output
	return soft_output

def continue_condition(soft_input, soft_output, iteration, cv, m_t, loss, labels):
	condition = (iteration < num_iterations)
	return condition

def belief_propagation_iteration(soft_input, soft_output, iteration, cv, m_t, loss, labels):
	# compute vc
	vc = compute_vc(cv,iteration,soft_input)

	# filter vc
	if decoder.relaxed:
		m_t = R * m_t + (1-R) * vc
		vc_prime = m_t
	else:
		vc_prime = vc

	# compute cv
	cv = compute_cv(vc_prime,iteration)

	# get output for this iteration
	soft_output = marginalize(soft_input, iteration, cv)
	iteration += 1

	# L = 0.5
	print("L = " + str(L))
	CE_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=-soft_output, labels=labels)) / num_iterations
	syndrome_loss = tf.reduce_mean(tf.maximum(1. - syndrome(soft_output, code),0) ) / num_iterations
	new_loss = L * CE_loss + (1-L) * syndrome_loss
	loss = loss + new_loss

	return soft_input, soft_output, iteration, cv, m_t, loss, labels

# builds a belief propagation TF graph
def belief_propagation_op(soft_input, labels):
	return tf.while_loop(
		continue_condition, # iteration < max iteration?
		belief_propagation_iteration, # compute messages for this iteration
		[
			soft_input, # soft input for this iteration
			soft_input,  # soft output for this iteration
			tf.constant(0,dtype=tf.int32), # iteration number
			tf.zeros([num_edges,batch_size],dtype=tf.float32), # cv
			tf.zeros([num_edges,batch_size],dtype=tf.float32), # m_t
			tf.constant(0.0,dtype=tf.float32), # loss
			labels
		]
		)

#### end decoder functions ####
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.01
learning_rate = starter_learning_rate # provided_decoder_type="normal", "FNNMS", "FNOMS", ...
decoder = Decoder(decoder_type=provided_decoder_type, random_seed=1, learning_rate = learning_rate, relaxed = False)
print("\n\nDecoder type: " + decoder.decoder_type + "\n\n")
if decoder.relaxed: print("relaxed")
else: print("not relaxed")

if SUM_PRODUCT:
	if decoder.decoder_type == "FNSPA":
		decoder.W_cv = tf.Variable(tf.truncated_normal([num_iterations, num_edges],dtype=tf.float32,stddev=1.0, seed=decoder.random_seed))
		
	if decoder.decoder_type == "RNSPA":
		decoder.W_cv = tf.Variable(tf.truncated_normal([num_edges],dtype=tf.float32,stddev=1.0, seed=decoder.random_seed))#tf.Variable(0.0,dtype=tf.float32)#
		
if MIN_SUM:
	if decoder.decoder_type == "FNNMS":
		# decoder.W_cv = tf.nn.softplus(tf.Variable(tf.truncated_normal([num_iterations, num_edges],dtype=tf.float32,stddev=1.0, seed=decoder.random_seed)))
		decoder.W_cv = tf.Variable(tf.truncated_normal([num_iterations, num_edges],dtype=tf.float32,stddev=1.0, seed=decoder.random_seed))
		
	if decoder.decoder_type == "FNOMS":
		decoder.B_cv = tf.Variable(tf.truncated_normal([num_iterations, num_edges],dtype=tf.float32,stddev=1.0))#tf.Variable(1.0 + tf.truncated_normal([num_iterations, num_edges],dtype=tf.float32,stddev=1.0))#tf.Variable(1.0 + tf.truncated_normal([num_iterations, num_edges],dtype=tf.float32,stddev=1.0/num_edges))

	if decoder.decoder_type == "RNNMS":
		decoder.W_cv = tf.nn.softplus(tf.Variable(tf.truncated_normal([num_edges],dtype=tf.float32,stddev=1.0, seed=decoder.random_seed)))#tf.Variable(0.0,dtype=tf.float32)#
		
	if decoder.decoder_type == "RNOMS":
		decoder.B_cv = tf.Variable(tf.truncated_normal([num_edges],dtype=tf.float32,stddev=1.0)) #tf.Variable(0.0,dtype=tf.float32)#

if decoder.relaxed:
	decoder.relaxation_factors = tf.Variable(0.0,dtype=tf.float32)
	R = tf.sigmoid(decoder.relaxation_factors)
	# print "single learned relaxation factor"

	# decoder.relaxation_factors = tf.Variable(tf.truncated_normal([num_edges],dtype=tf.float32,stddev=1.0))
	# R = tf.tile(tf.reshape(tf.sigmoid(decoder.relaxation_factors),[-1,1]),[1,batch_size])
	# print "multiple relaxation factors"

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
config = tf.ConfigProto(
        device_count = {'CPU': 2,'GPU': 0}
    )
with tf.Session(config=config) as session: #tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
	# simulate each SNR
	SNRs = np.arange(snr_lo, snr_hi+snr_step, snr_step)
	if (batch_size % len(SNRs)) != 0:
		print("********************")
		print("********************")
		print("error: batch size must divide by the number of SNRs to train on")
		print("********************")
		print("********************")
	BERs = []
	SERs = []
	FERs = []

	print("\nBuilding the decoder graph...")
	belief_propagation = belief_propagation_op(soft_input=tf_train_dataset, labels=tf_train_labels)
	if TRAINING:
		training_loss = belief_propagation[5]#tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=belief_propagation[1], labels=tf_train_labels))
		loss = training_loss
		print("Learning rate: " + str(starter_learning_rate))
		optimizer = tf.train.AdamOptimizer(learning_rate=decoder.learning_rate).minimize(loss,global_step=global_step)
	print("Done.\n")
	init = tf.global_variables_initializer()

	if ALL_ZEROS_CODEWORD_TRAINING:
		codewords = np.zeros([n,batch_size])
		codewords_repeated = np.zeros([num_iterations,n,batch_size]) # repeat for each iteration (multiloss)
		BPSK_codewords = np.ones([n,batch_size])
		soft_input = np.zeros_like(BPSK_codewords)
		channel_information = np.zeros_like(BPSK_codewords)

	covariance_matrix = np.eye(n)
	eta = 0.99
	for i in range(0,n):
		for j in range(0,n):
			covariance_matrix[i,j] = eta**np.abs(i-j)

	session.run(init)
	
	if TRAINING:
		# steps = 10001
		print("***********************")
		print("Training decoder using " + str(steps) + " minibatches...")
		print("***********************")

		step = 0
		while step < steps:
			# generate random codewords
			if not ALL_ZEROS_CODEWORD_TRAINING:
				# generate message
				messages = np.random.randint(0,2,[k,batch_size])

				# encode message
				codewords = np.dot(G, messages) % 2
				#codewords_repeated = np.tile(x,(num_iterations,1,1)).shape 

				# modulate codeword
				BPSK_codewords = (0.5 - codewords.astype(np.float32)) * 2.0

				soft_input = np.zeros_like(BPSK_codewords)
				channel_information = np.zeros_like(BPSK_codewords)
			else:
				codewords = np.zeros([n,batch_size])
				#codewords_repeated = np.zeros([num_iterations,n,batch_size]) # repeat for each iteration (multiloss)
				BPSK_codewords = np.ones([n,batch_size])
				soft_input = np.zeros_like(BPSK_codewords)
				channel_information = np.zeros_like(BPSK_codewords)

			# create minibatch with codewords from multiple SNRs
			for i in range(0,len(SNRs)):
				sigma = np.sqrt(1. / (2 * (np.float(k)/np.float(n)) * 10**(SNRs[i]/10)))
				noise = sigma * np.random.randn(n,batch_size//len(SNRs))
				start_idx = batch_size*i//len(SNRs)
				end_idx = batch_size*(i+1)//len(SNRs)
				channel_information[:,start_idx:end_idx] = BPSK_codewords[:,start_idx:end_idx] + noise
				if NO_SIGMA_SCALING_TRAIN:
					soft_input[:,start_idx:end_idx] = channel_information[:,start_idx:end_idx]
				else:
					soft_input[:,start_idx:end_idx] = 2.0*channel_information[:,start_idx:end_idx]/(sigma*sigma)


			# feed minibatch into BP and run SGD
			batch_data = soft_input
			batch_labels = codewords #codewords #codewords_repeated
			feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
			[_] = session.run([optimizer], feed_dict=feed_dict) #,bp_output,syndrome_output,belief_propagation, soft_syndromes

			if decoder.relaxed and TRAINING: 
				print(session.run(R))

			if step % 100 == 0:
				print(str(step) + " minibatches completed")

			step += 1
		
		print("Trained decoder on " + str(step) + " minibatches.\n")

	# testing phase
	print("***********************")
	print("Testing decoder...")
	print("***********************")
	for SNR in SNRs:
		# simulate this SNR
		sigma = np.sqrt(1. / (2 * (np.float(k)/np.float(n)) * 10**(SNR/10)))
		frame_count = 0
		bit_errors = 0
		frame_errors = 0
		frame_errors_with_HDD = 0
		symbol_errors = 0
		FE = 0

		# simulate frames
		while ((FE < min_frame_errors) or (frame_count < 100000)) and (frame_count < max_frames):
			frame_count += batch_size # use different batch size for test phase?

			if not ALL_ZEROS_CODEWORD_TESTING:
				# generate message
				messages = np.random.randint(0,2,[batch_size,k])

				# encode message
				codewords = np.dot(G, messages.transpose()) % 2

				# modulate codeword
				BPSK_codewords = (0.5 - codewords.astype(np.float32)) * 2.0

			# add Gaussian noise to codeword
			noise = sigma * np.random.randn(BPSK_codewords.shape[0],BPSK_codewords.shape[1])
			channel_information = BPSK_codewords + noise

			# convert channel information to LLR format
			if NO_SIGMA_SCALING_TEST:
				soft_input = channel_information
			else:
				soft_input = 2.0*channel_information/(sigma*sigma)

			# run belief propagation
			batch_data = soft_input
			feed_dict = {tf_train_dataset : batch_data, tf_train_labels : codewords}
			soft_outputs = session.run([belief_propagation], feed_dict=feed_dict)
			soft_output = np.array(soft_outputs[0][1])
			recovered_codewords = (soft_output < 0).astype(int)

			# update bit error count and frame error count
			errors = codewords != recovered_codewords
			bit_errors += errors.sum()
			frame_errors += (errors.sum(0) > 0).sum()

			FE = frame_errors

		# summarize this SNR:
		print("SNR: " + str(SNR))
		print("frame count: " + str(frame_count))

		bit_count = frame_count * n
		BER = np.float(bit_errors) / np.float(bit_count)
		BERs.append(BER)
		print("bit errors: " + str(bit_errors))
		print("BER: " + str(BER))

		FER = np.float(frame_errors) / np.float(frame_count)
		FERs.append(FER)
		print("FER: " + str(FER))
		print("")

	# print summary
	print("BERs:")
	print(BERs)
	print("FERs:")
	print(FERs)	

	# offset = session.run(decoder.B_cv)
	# weights = session.run(decoder.W_cv)
