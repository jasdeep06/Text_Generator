import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn




sequence_length=50
embedding_size=100
n_hidden=128
vocab_size=57
batch_size=20

f=open("dataset/Tyrion.txt")
text=f.read()
unique_chars=[]
for character in text:
    if character not in unique_chars:
        unique_chars.append(character)




characters=list(set(text))

character_index_dict={character:index for index,character in enumerate(characters) }
index_character_dict={index:character for index,character in enumerate(characters)}


def pair_char_int(seq,int_list):
    int_seq=[]
    for character in seq:
        index=unique_chars.index(character)
        int_seq.append(int_list[index])
    return int_seq


def generate_random_batches():
    begin_index=np.random.random_integers(len(text)-batch_size*sequence_length-4)
    #print(begin_index)

    batch_text_x=text[begin_index:begin_index+batch_size*sequence_length]
    batch_text_y=text[begin_index+1:begin_index+batch_size*sequence_length+1]
    return batch_text_x,batch_text_y



def process(batch):
    #batch_size=len(batch)//sequence_length
    targets=[]
    for i in range(batch_size):
        internal_list = []

        for character in batch[i*sequence_length:sequence_length*(i+1)]:

            internal_list.append(character_index_dict[character])
        targets.append(internal_list)
    return targets



graph=tf.Graph()

with graph.as_default():


    with tf.name_scope("training"):
        x=tf.placeholder(tf.int64,[None,sequence_length],name="x_placeholder")
        y=tf.placeholder(tf.int64,[None,sequence_length],name="y_placeholder")
        #state_holder=tf.placeholder("float",[2,1,128])

        """""
        c_state=tf.placeholder("float",[batch_size,n_hidden],name="c_placeholder")
        h_state=tf.placeholder("float",[batch_size,n_hidden],name="h_placeholder")
        """""
        h_state = tf.Variable(initial_value=tf.zeros([batch_size, n_hidden], dtype="float32"), trainable=False)
        c_state = tf.Variable(initial_value=tf.zeros([batch_size, n_hidden], dtype="float32"), trainable=False)



        out_weights=tf.get_variable(initializer=tf.random_normal([n_hidden,vocab_size]),name="out_weights")
        out_bias=tf.get_variable(name="out_bias",initializer=tf.random_normal([vocab_size]))

        '''''
targets=[]
for i in range(25):
    targets.append(tf.convert_to_tensor(np.random.randint(57,size=(3,))))
targets=tf.convert_to_tensor(targets)
    '''''
#int_seq=pair_char_int(seq,int_list)
#int_seq (3,25)
#int_seq=np.random.randint(57,size=(3,25))
        embedding=tf.get_variable(name="embedding",shape=[vocab_size,embedding_size])
#embedding_looked_up (3,25,100)
        embedding_looked_up=tf.nn.embedding_lookup(embedding,x,name="embedding_lookup")


#input=tf.reshape(embedding_looked_up,(3,seq_length,embedding_size))

        new_input=tf.unstack(embedding_looked_up,sequence_length,1)
#new_input is list of length 25 consisting of 25 arrays of shape (3,100)
        lstm_layer=rnn.BasicLSTMCell(n_hidden,forget_bias=1)

        initial_state=tf.nn.rnn_cell.LSTMStateTuple(c_state,h_state)
        outputs,state=rnn.static_rnn(lstm_layer,new_input,dtype="float32",initial_state=initial_state)
        assign_op_c=tf.assign(c_state,state[0])
        assign_op_h=tf.assign(h_state,state[1])
#outputs is list of 25 arrays of shape (3,128)


        out_shape=tf.reshape(outputs,[-1,n_hidden])
#out_shape is of shape (75,128)
        prediction_scores=tf.add(tf.matmul(out_shape,out_weights),out_bias,name="addition-0")
#prediction_scores is (75,57)
        logits=tf.reshape(prediction_scores,(batch_size,sequence_length,vocab_size))
#logits is (3,25,57)


        loss=tf.contrib.seq2seq.sequence_loss(logits=logits,targets=y,weights=tf.ones([batch_size, sequence_length]),average_across_timesteps=False,
        average_across_batch=True)
#print(sess.run(prediction_scores))
#prediction=tf.argmax(prediction_scores)
        net_loss=tf.reduce_sum(loss)

        optimization=tf.train.GradientDescentOptimizer(learning_rate=.00001).minimize(net_loss)



        saver=tf.train.Saver(tf.trainable_variables())

        for v in tf.all_variables():
            print(v.name)

        print("over")

    #with tf.name_scope("prediction"):
        tf.get_variable_scope().reuse_variables()
        x_pred=tf.placeholder(tf.int64,shape=[1,1])

        """""
        c_state_pred = tf.placeholder("float", [1, n_hidden], name="c_placeholder")
        h_state_pred = tf.placeholder("float", [1, n_hidden], name="h_placeholder")
        """""
        c_state_pred=tf.Variable(initial_value=tf.zeros([1,n_hidden],"float32"))
        h_state_pred=tf.Variable(initial_value=tf.zeros([1,n_hidden],"float32"))


        out_weights_predict = tf.get_variable( name="out_weights")
        out_bias_predict = tf.get_variable(name="out_bias")
        embedding_predict=tf.get_variable(name="embedding")

        embedding_looked_up_predict = tf.nn.embedding_lookup(embedding_predict, x_pred, name="embedding_lookup_predict")

        new_input_predict=tf.unstack(embedding_looked_up_predict,1,1)

        lstm_layer_predict = rnn.BasicLSTMCell(n_hidden, forget_bias=1)
        initial_state_predict = tf.nn.rnn_cell.LSTMStateTuple(c_state_pred, h_state_pred)
        outputs_predict, state_predict = rnn.static_rnn(lstm_layer_predict, new_input_predict, dtype="float32",initial_state=initial_state_predict)
        assign_op_c_predict=tf.assign(c_state_pred,state_predict[0])
        assign_op_h_predict=tf.assign(h_state_pred,state_predict[1])



        out_shape_predict = tf.reshape(outputs_predict, [-1, n_hidden])
        prediction_scores_predict = tf.add(tf.matmul(out_shape_predict, out_weights_predict), out_bias_predict, name="addition-0-predict")
        predicted_char=tf.argmax(tf.nn.softmax(prediction_scores_predict),axis=1)
        for v in tf.all_variables():
            print(v.name)









with tf.Session(graph=graph) as sess:
    #initializes values of variables in graph
    sess.run(tf.global_variables_initializer())

    next_state=[]
    iter=1
    current_index=0
    writer = tf.summary.FileWriter("log/", graph=graph)
    while iter<500:
        #print("iter ",iter)


        batch_x,batch_y=generate_random_batches()


        #print(batch_x)
        batch_y=process(batch_y)
        batch_x=process(batch_x)

        #print(batch_x)
        #print(batch_y)
        #sess.run(optimization,feed_dict={x:batch_x,y:batch_y})
        """""

        if iter==1:
           #print(sess.run(state,feed_dict={x:batch_x,y:batch_y,c_state:np.zeros((2,128),"float"),h_state:np.zeros((2,128),"float")}))
           next_state= sess.run(state,feed_dict={x:batch_x,y:batch_y,c_state:np.zeros((batch_size,n_hidden),"float"),h_state:np.zeros((batch_size,n_hidden),"float")})
           sess.run(optimization,feed_dict={x:batch_x,y:batch_y,c_state:np.zeros((batch_size,n_hidden),"float"),h_state:np.zeros((batch_size,n_hidden),"float")})
           #print(next_state)
           #print(next_state)

        if iter>1:
            #print(sess.run(loss,feed_dict={x:batch_x,y:batch_y,c_state:next_state[0],h_state:next_state[1]}))
            #print(next_state)


            print(sess.run(net_loss,feed_dict={x:batch_x,y:batch_y,c_state:next_state[0],h_state:next_state[1]}))
            sess.run(optimization,feed_dict={x:batch_x,y:batch_y,c_state:next_state[0],h_state:next_state[1]})
            next_state = sess.run(state,
                                  feed_dict={x: batch_x, y: batch_y, c_state: next_state[0], h_state: next_state[1]})
        """""



        sess.run(optimization,feed_dict={x:batch_x,y:batch_y})
        #print(sess.run(tf.get_default_graph().get_tensor_by_name('rnn/basic_lstm_cell/bias:0'),feed_dict={x:batch_x,y:batch_y}))

        print(sess.run(net_loss,feed_dict={x:batch_x,y:batch_y}))
        sess.run(assign_op_c,feed_dict={x:batch_x,y:batch_y})
        sess.run(assign_op_h,feed_dict={x:batch_x,y:batch_y})





        """""
        if iter%10==0:
            print("iteration ",iter)
            los = sess.run(net_loss, feed_dict={x: batch_x, y: batch_y})
            print("loss ", los)
        if iter==999:
            saver.save(sess,"saved/my_model-0")
            print(sess.run(state,feed_dict={x: batch_x, y: batch_y}))
        """""

        iter=iter+1



    seed_char = 'A'
    seed_index = character_index_dict[seed_char]
    print(character_index_dict)
    pred_char = 1


    while pred_char<20:

        """""

            if pred_char==1:

                next_state_predict = sess.run(state_predict,
                                    feed_dict={x_pred: [[seed_index]], c_state_pred: np.zeros((1, n_hidden), "float"),
                                                h_state_pred: np.zeros((1, n_hidden), "float")})

                char=sess.run(predicted_char,feed_dict={x_pred: [[seed_index]], c_state_pred: np.zeros((1, n_hidden), "float"),
                                             h_state_pred: np.zeros((1, n_hidden), "float")})


                predicted=index_character_dict[char[0]]
                seed_index=char[0]
                output.append(predicted)
                print(predicted)

            if pred_char>1:

                char = sess.run(predicted_char,
                                feed_dict={x_pred: [[seed_index]], c_state_pred:next_state_predict[0],
                                       h_state_pred: next_state_predict[1]})


                predicted = index_character_dict[char[0]]
                seed_index = char[0]
                next_state_predict = sess.run(state_predict,
                                          feed_dict={x_pred: [[seed_index]],
                                                     c_state_pred: next_state_predict[0],
                                                     h_state_pred: next_state_predict[1]})
                output.append(predicted)
                print(predicted)






        big_output.append(output)

    """""
        #print(seed_index)
        #print(sess.run(initial_state_predict,feed_dict={x_pred:[[seed_index]]}))
        #print(sess.run(tf.get_default_graph().get_tensor_by_name('rnn/basic_lstm_cell/kernel:0'),feed_dict={x_pred:[[seed_index]]}))

        char=sess.run(predicted_char,feed_dict={x_pred:[[seed_index]]})
        print(sess.run(prediction_scores_predict,feed_dict={x_pred:[[seed_index]]}))
        sess.run(assign_op_c_predict,feed_dict={x_pred:[[seed_index]]})
        sess.run(assign_op_h_predict,feed_dict={x_pred:[[seed_index]]})
        predicted = index_character_dict[char[0]]
        seed_index = char[0]
        print(predicted)
        pred_char = pred_char + 1







'''''
with tf.Session() as sample_sess:
    new_saver=tf.train.import_meta_graph("saved/my_model-0.meta")
    new_saver.restore(sample_sess, "saved/model-0")
    print(sess.run("addition-0",feed_dict=)


'''''












