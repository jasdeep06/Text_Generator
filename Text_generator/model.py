import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

class Model():
    def __init__(self,sequence_length,n_hidden,vocab_size,batch_size,embedding_size,num_layers,keep_prob,learning_rate,training=None,resume_training=None):
        self.sequence_length=sequence_length
        self.n_hidden=n_hidden
        self.vocab_size=vocab_size
        self.batch_size=batch_size
        self.embedding_size=embedding_size
        self.num_layers=num_layers
        self.learning_rate=learning_rate
        self.training=training
        self.keep_prob=keep_prob
        self.resume_training=resume_training


        if self.training==False:
            self.sequence_length=1
            self.batch_size=1
            self.keep_prob=1
            reuse=True
        else:
            reuse=None







        with tf.variable_scope("model-scope"):

            self.x=tf.placeholder(tf.int64,[None,self.sequence_length],"x_placeholder")
            self.y=tf.placeholder(tf.int64,[None,self.sequence_length],"y_placeholder")
            self.h_state = tf.placeholder(tf.float32, [self.num_layers, self.batch_size, self.n_hidden],name="h_state")
            self.c_state = tf.placeholder(tf.float32, [self.num_layers, self.batch_size, self.n_hidden],name="t_state")

            with tf.variable_scope("softmax",reuse=reuse):

                self.out_weight=tf.get_variable(initializer=tf.random_normal([self.n_hidden,self.vocab_size]),name="out_weight")
                out_bias=tf.get_variable(initializer=tf.random_normal([self.vocab_size]),name="out_bias")




            unstacked_h_state = tf.unstack(self.h_state, num_layers, axis=0)
            unstacked_c_state = tf.unstack(self.c_state, num_layers, axis=0)



            with tf.variable_scope("embedding",reuse=reuse):
                embedding=tf.get_variable(name="embedding",shape=[self.vocab_size,self.embedding_size])
            embedding_looked_up = tf.nn.embedding_lookup(embedding, self.x, name="embedding_lookup")

            new_input = tf.unstack(embedding_looked_up, self.sequence_length, 1)




            cell = tf.contrib.rnn.MultiRNNCell(
                [rnn.DropoutWrapper(rnn.BasicLSTMCell(self.n_hidden), output_keep_prob=self.keep_prob) for _ in range(self.num_layers)],
                state_is_tuple=True)


            self.initial_state = []
            for i in range(num_layers):
                self.initial_state.append(tf.nn.rnn_cell.LSTMStateTuple(unstacked_c_state[i], unstacked_h_state[i]))
            with tf.variable_scope("lstm",reuse=reuse):

                outputs, self.state = rnn.static_rnn(cell, new_input, dtype="float32", initial_state=self.initial_state)



            out_shape = tf.reshape(outputs, [-1, self.n_hidden])
            prediction_scores = tf.add(tf.matmul(out_shape, self.out_weight), out_bias, name="addition-0")
            self.prediction_probab=tf.nn.softmax(prediction_scores)
            logits = tf.reshape(prediction_scores, (self.batch_size, self.sequence_length, self.vocab_size))


            if self.training==True:
                self.net_loss = tf.contrib.seq2seq.sequence_loss(logits=logits, targets=self.y,
                                                weights=tf.ones([self.batch_size, self.sequence_length]),
                                                average_across_timesteps=True,
                                                average_across_batch=True)

                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                trainable_variables=tf.trainable_variables()
                self.gradients=tf.gradients(self.net_loss,trainable_variables)
                self.global_step=tf.Variable(0,dtype=tf.int32,trainable=False,name="global_step")
                self.train_op=optimizer.apply_gradients(zip(self.gradients,trainable_variables),name="train_op",global_step=self.global_step)















