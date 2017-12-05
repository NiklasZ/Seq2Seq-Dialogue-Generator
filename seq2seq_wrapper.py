import tensorflow as tf
import numpy as np
import sys
import data_utils
import time
import os
import load_embeddings
import legacy_seq2seq

BASELINE_MODE = 0
ATTENTION_MODE = 1
ATTENTION_DIVERSITY_MODE = 2


class Seq2Seq(object):

    def __init__(self,
                 xseq_len,
                 yseq_len,
                 xvocab_size,
                 yvocab_size,
                 emb_dim,
                 num_layers,
                 ckpt_path,
                 lr=0.0001,
                 epochs=40000,
                 mode=BASELINE_MODE,
                 model_name='seq2seq_model',
                 p_r=None):

        # attach these arguments to self
        self.xseq_len = xseq_len
        self.yseq_len = yseq_len
        self.ckpt_path = ckpt_path
        self.epochs = epochs
        self.model_name = model_name
        self.mode = mode
        self.batch_n = 1

        # build thy graph
        #  attach any part of the graph that needs to be exposed, to the self
        def __graph__():

            # placeholders
            tf.reset_default_graph()
            #  encoder inputs : list of indices of length xseq_len
            self.enc_ip = [
                tf.placeholder(
                    shape=[
                        None,
                    ], dtype=tf.int64, name='ei_{}'.format(t))
                for t in range(xseq_len)
            ]

            #  labels that represent the real outputs
            self.labels = [
                tf.placeholder(
                    shape=[
                        None,
                    ], dtype=tf.int64, name='ei_{}'.format(t))
                for t in range(yseq_len)
            ]

            #  decoder inputs : 'GO' + [ y1, y2, ... y_t-1 ]
            self.dec_ip = [
                tf.zeros_like(self.enc_ip[0], dtype=tf.int64, name='GO')
            ] + self.labels[:-1]

            # Basic LSTM cell wrapped in Dropout Wrapper
            self.keep_prob = tf.placeholder(tf.float32)
            # define the basic cell
            basic_cell = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(
                tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(
                    emb_dim, state_is_tuple=True),
                output_keep_prob=self.keep_prob)
            # stack cells together : n layered model
            stacked_lstm = tf.contrib.rnn.core_rnn_cell.MultiRNNCell(
                [basic_cell] * num_layers, state_is_tuple=True)

            # for parameter sharing between training model
            #  and testing model
            with tf.variable_scope('decoder') as scope:

                args = {'encoder_inputs': self.enc_ip,
                        'decoder_inputs': self.dec_ip,
                        'cell': stacked_lstm,
                        'num_encoder_symbols': xvocab_size,
                        'num_decoder_symbols': yvocab_size,
                        'embedding_size': emb_dim}

                if self.mode == BASELINE_MODE:
                    model = legacy_seq2seq.embedding_rnn_seq2seq
                elif self.mode == ATTENTION_MODE or \
                     self.mode == ATTENTION_DIVERSITY_MODE:
                    model = \
                        legacy_seq2seq.embedding_attention_seq2seq
                    projection_w = tf.get_variable(
                        name='projection_weights',
                        shape=[emb_dim, yvocab_size],
                        initializer=tf.contrib.layers.xavier_initializer())
                    projection_b = tf.get_variable(
                        name='projection_biases',
                        shape=[yvocab_size, ])
                    proj_tup = (projection_w, projection_b)
                    args.update({'output_projection': proj_tup})
                    args.update({'p_r': p_r})

                # build the seq2seq model
                #  inputs : encoder, decoder inputs, LSTM cell type,
                # vocabulary sizes, embedding dimensions
                self.decode_outputs, self.decode_states = model(**args)

                # share parameters
                scope.reuse_variables()

                # testing model, where output of previous timestep is fed as
                # input to the next timestep
                args.update({'feed_previous': True})
                self.decode_outputs_test, self.decode_states_test = \
                    model(**args)

            if self.mode == ATTENTION_MODE or \
               self.mode == ATTENTION_DIVERSITY_MODE:
                self.decode_outputs = [
                    tf.matmul(output, projection_w) + projection_b
                    for output in self.decode_outputs
                ]
                self.decode_outputs_test = [
                    tf.matmul(output, projection_w) + projection_b
                    for output in self.decode_outputs_test
                ]

            # now, for training,
            #  build loss function

            # weighted loss
            #  TODO : add parameter hint
            loss_weights = [
                tf.ones_like(label, dtype=tf.float32) for label in self.labels
            ]
            self.loss = tf.contrib.legacy_seq2seq.sequence_loss(
                self.decode_outputs, self.labels, loss_weights, yvocab_size)
            # train op to minimize the loss
            self.train_op = tf.train.AdamOptimizer(
                learning_rate=lr).minimize(self.loss)

            self.train_loss_summary = tf.summary.scalar("train_loss", self.loss)
            self.validation_loss_summary = tf.summary.scalar("validation_loss", self.loss)

        timestamp = int(time.time())
        run_log_dir = os.path.join('summaries',
                                   'vanilla' + '_' + str(timestamp))
        os.makedirs(run_log_dir)
        # (this step also writes the graph to the events file so that
        # it shows up in TensorBoard)
        self.summary_writer = tf.summary.FileWriter(run_log_dir, flush_secs=5)

        # build comput graph
        __graph__()

    '''
        Training and Evaluation

    '''

    # get the feed dictionary
    def get_feed(self, X, Y, keep_prob):
        feed_dict = {self.enc_ip[t]: X[t] for t in range(self.xseq_len)}
        if Y is not None:
            feed_dict.update({self.labels[t]: Y[t] for t in range(self.yseq_len)})
        feed_dict[self.keep_prob] = keep_prob  # dropout prob
        return feed_dict

    def sample_replies(self, sess, valid_set, metadata, batch_n):
        test_x = valid_set.__next__()[0]
        test_y_pred = self.predict(sess, test_x)

        log_file = open('logs/%d.txt' % batch_n, 'w')

        replies = []
        for ii, oi in zip(test_x.T, test_y_pred):
            q = data_utils.decode(sequence=ii, lookup=metadata['idx2w'], separator=' ')
            decoded = data_utils.decode(sequence=oi, lookup=metadata['idx2w'], separator=' ')
            if decoded not in replies:
                log_file.write('q: "%s"; a: "%s"\n' % (q, decoded))
                replies.append(decoded)

        log_file.write('%d/%d\n' % (len(replies), test_x.shape[1]))

        log_file.close()

    # run one batch for training
    def train_batch(self, sess, train_batch_gen, step):
        # get batches
        batchX, batchY = train_batch_gen.__next__()
        # build feed
        feed_dict = self.get_feed(batchX, batchY, keep_prob=0.5)
        _, loss_v, s_v = sess.run(
            [self.train_op, self.loss, self.train_loss_summary], feed_dict)
        self.summary_writer.add_summary(s_v, step)
        return loss_v

    def eval_step(self, sess, eval_batch_gen, step, save):
        # get batches
        batchX, batchY = eval_batch_gen.__next__()
        # build feed
        feed_dict = self.get_feed(batchX, batchY, keep_prob=1.)
        loss_v, dec_op_v, s_v = sess.run(
            [self.loss, self.decode_outputs_test,
             self.validation_loss_summary], feed_dict)
        if save:
            self.summary_writer.add_summary(s_v, step)
        # dec_op_v is a list; also need to transpose 0,1 indices
        #  (interchange batch_size and timesteps dimensions
        dec_op_v = np.array(dec_op_v).transpose([1, 0, 2])
        return loss_v, dec_op_v, batchX, batchY

    # evaluate 'num_batches' batches
    def eval_batches(self, sess, eval_batch_gen, num_batches, step):
        losses = []
        for i in range(num_batches):
            if i == 0:
                save = True
            else:
                save = False
            loss_v, dec_op_v, batchX, batchY = self.eval_step(
                sess, eval_batch_gen, step, save)
            losses.append(loss_v)
        return np.mean(losses)

    @staticmethod
    def load_embeddings(sess, model, metadata, path):
        tv_dict = {
            name: variable
            for name, variable in [(v.name, v)
                                   for v in tf.trainable_variables()]
        }
        if model.mode == ATTENTION_MODE or \
           model.mode == ATTENTION_DIVERSITY_MODE:
            name1 = 'embedding_attention_seq2seq'
            name2 = 'embedding_attention_decoder'
        else:
            name1 = 'embedding_rnn_seq2seq'
            name2 = 'embedding_rnn_decoder'
        emb1 = tv_dict[
            'decoder/%s/rnn/embedding_wrapper/embedding:0' % name1]
        emb2 = tv_dict[
            'decoder/%s/%s/embedding:0' % (name1, name2)]
        emb1_val = sess.run(emb1)
        emb2_val = sess.run(emb2)
        print(np.sum(emb1_val), np.sum(emb2_val))
        load_embeddings.load_embedding(
            sess,
            metadata['w2idx'], [emb1, emb2],
            path,
            dim_embedding=300,
            vocab_length=len(metadata['w2idx']))
        emb1_val = sess.run(emb1)
        emb2_val = sess.run(emb2)
        print(np.sum(emb1_val), np.sum(emb2_val))

    # finally the train function that
    #  runs the train_op in a session
    #   evaluates on valid set periodically
    #    prints statistics
    def train(self, train_set, valid_set, metadata, sess=None):

        # we need to save the model periodically
        saver = tf.train.Saver()

        # if no session is given
        if not sess:
            # create a session
            sess = tf.Session()
            # init all variables
            sess.run(tf.global_variables_initializer())

        sys.stdout.write('\n<log> Training started </log>\n')
        start_time = time.time()
        # run M epochs
        while self.batch_n < self.epochs:
            bs = time.time()
            print("Batch %d" % self.batch_n)
            try:
                self.train_batch(sess, train_set, self.batch_n)
                if self.batch_n % 100 == 0:  # TODO : make this tunable by the user
                    val_loss = self.eval_batches(sess, valid_set, 1, self.batch_n)
                    print('val   loss : {0:.6f}'.format(val_loss))
                    sys.stdout.flush()
                    self.sample_replies(sess, valid_set, metadata, self.batch_n)
            except KeyboardInterrupt:  # this will most definitely happen, so handle it
                print('Interrupted by user at iteration {}'.format(self.batch_n))
                self.session = sess
                return sess

            cur_time = time.time()
            seconds_per_batch = (cur_time - start_time) / self.batch_n
            print("Average seconds per batch: %.2f" % seconds_per_batch)
            print("Time for this batch: %.2f" % (cur_time - bs))
            seconds_remaining = (self.epochs - self.batch_n) * seconds_per_batch
            minutes_remaining = seconds_remaining / 60
            print("ETA until finish: %d minutes" % minutes_remaining)
            eta2 = (self.epochs - self.batch_n) * (cur_time - bs) / 60
            print("eta 2: %d minutes" % eta2)

            self.batch_n += 1

    def restore_last_session(self):
        saver = tf.train.Saver()
        # create a session
        sess = tf.Session()
        # get checkpoint state
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
        # restore session
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        # return to user
        return sess

    # prediction
    def predict(self, sess, X, Y=None, argmax=True):
        feed_dict = self.get_feed(X, Y, keep_prob=1.)
        if Y is None:
            op = self.decode_outputs_test
        else:
            op = self.decode_outputs
        dec_op_v = sess.run(op, feed_dict)
        # dec_op_v is a list; also need to transpose 0,1 indices
        #  (interchange batch_size and timesteps dimensions
        dec_op_v = np.array(dec_op_v).transpose([1, 0, 2])
        if argmax:
            # return the index of item with highest probability
            return np.argmax(dec_op_v, axis=2)
        else:
            return dec_op_v
