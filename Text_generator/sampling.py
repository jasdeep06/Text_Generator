import tensorflow as tf
from input_generator import InputGenerator
from model import Model
from state_handler import make_state
import numpy as np


input_generator=InputGenerator(file_name="dataset/shakespeare.txt",batch_size=50,sequence_length=50)
vocab_size,text=input_generator.get_vocab_size()
character_to_index=input_generator.char_to_index()
index_to_character=input_generator.index_to_char(character_to_index)


def sample(iteration=0,checkpoint_dir=""):

    model = Model(sequence_length=50,
                  n_hidden=128,
                  vocab_size=vocab_size,
                  batch_size=50,
                  embedding_size=100,
                  num_layers=2, keep_prob=0.5, learning_rate=0.01, training=False)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(checkpoint_dir)


        saver=tf.train.Saver(tf.global_variables())

        #ckpt=tf.train.get_checkpoint_state("trial/")
        #saver.restore(sess,ckpt.model_checkpoint_path)
        saver.restore(sess,checkpoint_dir)







        seed='The '
        pred_char=0
        output=""
        seed_index = character_to_index[seed[pred_char]]

        next_state=sess.run(model.initial_state,feed_dict={model.c_state:np.zeros((model.num_layers,model.batch_size,model.n_hidden),"float"),model.h_state:np.zeros((model.num_layers,model.batch_size,model.n_hidden),"float")})

        while pred_char<250:

            new_c,new_h=make_state(model,next_state)
            _, next_state, probab = sess.run(
                [model.initial_state, model.state, model.prediction_probab], feed_dict={
                    model.c_state: new_c,
                    model.h_state: new_h, model.x: [[seed_index]]
                    })


            if pred_char<len(seed):
                seed_index=character_to_index[seed[pred_char]]

            else:
                predicted_char_index = tf.argmax(probab, axis=1)

                seed_index = sess.run(predicted_char_index)[0]
                predicted_char = index_to_character[seed_index]
                output = output + predicted_char


            pred_char=pred_char+1



        with open("output/output.txt", 'a') as f:
            f.write("\n"+str(iteration)+"\n"+output)





def main():
    sample()

if __name__ == '__main__':
    main()


