from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ptb_reader
import numpy as np
import os
import tensorflow as tf
import traceback


flags = tf.flags
flags.DEFINE_string('data_dir', '../data/', 'Directory containing PTB data')
flags.DEFINE_string('batchsize', 30, 'Mini-batch Size')
flags.DEFINE_string('numsteps', 20, 'Number of steps of recurrence (size of context window ie. how many words from the history to depend on)')
flags.DEFINE_string('numlayers', 2, 'Number of LSTM Layers')
flags.DEFINE_string('vocab_size', 10000, 'Size of vocabulary')
flags.DEFINE_string('embdim', 100, 'Size of Word Embeddings (hidden size of LSTM Cell, note: 1 LSTM Cell -> 1 Word -> "embdim"-sized embedding vector')
flags.DEFINE_string('max_norm', 5, 'Max Norm of gradient after which it will be clipped')
flags.DEFINE_string('max_max_epoch', 10, 'No. of times the LSTM cell state and value are reinitialized and trained on a stream of input')
flags.DEFINE_string('log_dir', '../lm-logs/', 'Directory for saving logs')
flags = flags.FLAGS

learning_rate = 1.0
decay = 0.5


def makemodel(is_training):
	X = tf.placeholder(tf.int32, [flags.batchsize, flags.numsteps], name='input')
	Y = tf.placeholder(tf.int32, [flags.batchsize, flags.numsteps], name='target')
	E = tf.get_variable('emb', [flags.vocab_size, flags.embdim], dtype=tf.float32)

	inp = tf.nn.embedding_lookup(E, X)

	lstm_cell = tf.contrib.rnn.LSTMCell(flags.embdim, state_is_tuple=True)
	lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=0.9)
	cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * flags.numlayers)
	state = cell.zero_state(flags.batchsize, tf.float32)
	initial_state = state

	lstm_output, final_state = tf.nn.dynamic_rnn(cell, inp, initial_state=initial_state)
	formatted_lstm_ouput = tf.reshape(lstm_output, [-1, flags.numsteps, flags.embdim])
	outputs = tf.transpose(formatted_lstm_ouput, [1, 0, 2])

	model_outputs = []
	for i in range(flags.numsteps):
		model_outputs.append(tf.matmul(outputs[i], tf.transpose(E)))
	model_outputs = tf.stack(model_outputs)
	model_outputs = tf.transpose(model_outputs, [1, 0, 2])

	loss = tf.contrib.seq2seq.sequence_loss(model_outputs, Y, tf.ones(tf.shape(Y)))
	lr = tf.Variable(0.0, trainable=False)

	if is_training:
		train_vars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(loss, train_vars), 5)
		opt = tf.train.GradientDescentOptimizer(learning_rate = lr)
		optimizer = opt.apply_gradients(zip(grads, train_vars), global_step=tf.contrib.framework.get_or_create_global_step())
	else:
		optimizer = tf.no_op()
	
	modeldict = {'X': X, 'Y': Y, 'initial_state': initial_state, 'final_state': final_state, 'loss': loss, 'optimizer': optimizer, 'lr': lr}
	return modeldict


def runepoch(sess, data, modeldict, fetches, epoch_no, verbose):
		lr_decay = decay ** max(epoch_no - 4, 0.0)
		sess.run(tf.assign(modeldict['lr'], learning_rate * lr_decay))
		state = sess.run(modeldict['initial_state'])
		losses = 0.0
		itercnt = 0

		if verbose: print('Running New Epoch')

		for curr, (x, y) in enumerate(ptb_reader.ptb_iterator(data, flags.batchsize, flags.numsteps)):
			feed_dict = {modeldict['X']: x, modeldict['Y']: y, modeldict['initial_state']: state}
			
			run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
			run_metadata = tf.RunMetadata()
			vals = sess.run(fetches, feed_dict)
			losses += vals['loss'] * flags.numsteps
			state = vals['final_state']
			itercnt += flags.numsteps
			if curr % 100 == 0 and verbose: print('Curr: ', curr, ' | Perplexity: ', np.exp(losses / itercnt))

		if verbose: print('Epoch Complete')
		return np.exp(losses / itercnt)


def train(train_data):
	with tf.Session() as sess:
		m_train = makemodel(True)
		print('Model Creation Done')
		
		fetches = dict()
		fetches['final_state'] = m_train['final_state']
		fetches['loss'] = m_train['loss']
		fetches['optimizer'] = m_train['optimizer']
		
		print('Training Started')
		saver = tf.train.Saver(max_to_keep=1)
		tf.train.write_graph(sess.graph_def, 'weights/', 'lstm-langModel.pb', as_text=False)
		sess.run(tf.global_variables_initializer())
		for i in range(flags.max_max_epoch):
			runepoch(sess, train_data, m_train, fetches, i, True)
			save_path = saver.save(sess, 'weights/lstm-langModel-checkpoint' + str(i))
			print("Model saved in file: %s" % save_path)
		save_path = saver.save(sess, 'weights/lstm-langModel')
		print('Training Complete')


def test(test_data):
	with tf.Session() as sess:
		m_test = makemodel(False)		
		fetches = dict()
		fetches['final_state'] = m_test['final_state']
		fetches['loss'] = m_test['loss']		
		saver = tf.train.Saver(max_to_keep=1)
		saver.restore(sess, 'weights-11/lstm-langModel')
		sess.run(tf.global_variables_initializer())
		p = runepoch(sess, test_data, m_test, fetches, 0, True)
		print(p)


def download_weights():
	if not os.path.exists('weights'):
		print('Downloading model from my github repository ...')
		os.system('wget -P weights https://raw.github.com/jatinarora2702/DeepLearning/master/assign4/13CS10057_Jatin/weights/')
		print('Download Complete.')


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--test", nargs = 1)
	args = parser.parse_args()
	try:
		if len(args.test) > 0:
			test_file = args.test[0]
			test_data, _ = ptb_reader.custom_raw_data(test_file)
			download_weights()
			test(test_data)
		else:
			train_data, valid_data, test_data, train_vocab_size = ptb_reader.ptb_raw_data(flags.data_dir)
			test(test_data)
	except:
		traceback.print_exc()
		train_data, valid_data, test_data, train_vocab_size = ptb_reader.ptb_raw_data(flags.data_dir)
		train(train_data)


if __name__ == '__main__':
	main()
