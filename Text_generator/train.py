import tensorflow as tf
from model import Model
import numpy as np
from input_generator import InputGenerator
import pickle
from state_handler import make_state
from sampling import sample

input_generator=InputGenerator(file_name="dataset/shakespeare.txt",batch_size=50,sequence_length=50)
vocab_size,text=input_generator.get_vocab_size()
character_to_index=input_generator.char_to_index()
index_to_character=input_generator.index_to_char(character_to_index)

def train(resume=None,intermediate_sampling=0):
    model = Model(sequence_length=50,
                  n_hidden=128,
                 vocab_size=vocab_size,
                batch_size=50,
               embedding_size=100,
              num_layers=2, keep_prob=0.5, learning_rate=0.01, training=True)










    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        iteration=0
        epoch=0
        num_iter=1000



        if resume==True:
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state("trial/")
            saver.restore(sess, ckpt.model_checkpoint_path)
            print(sess.run(model.out_weight))
            with open('save_state.pkl', 'rb') as f:
                next_state, begin_index = pickle.load(f)
                print(next_state)
        else:

            begin_index=0

            next_state=sess.run(model.initial_state,feed_dict={model.c_state:np.zeros((model.num_layers,model.batch_size,model.n_hidden),"float"),model.h_state:np.zeros((model.num_layers,model.batch_size,model.n_hidden),"float")})




        while True:




            if begin_index>(len(text) - model.batch_size * model.sequence_length - 4):
                begin_index=0
                epoch=epoch+1
                print("Epoch completed")



            batch_x, batch_y = input_generator.generate_ordered_batches(begin_index)


            begin_index=begin_index+model.batch_size*model.sequence_length




            batch_x = input_generator.process_batches(batch_x, character_to_index)
            batch_y = input_generator.process_batches(batch_y, character_to_index)





            new_c,new_h=make_state(model,next_state)
            _, next_state,gradient, _ ,loss= sess.run([model.initial_state, model.state,model.gradients, model.train_op,model.net_loss], feed_dict={
                model.c_state: new_c,
                model.h_state: new_h, model.x: batch_x,
                model.y: batch_y})


            print("iteration ",iteration)
            print("epoch ",epoch)

            print("loss ",loss)
            print(
                    "_________________________________________________________________________________________________")



            if intermediate_sampling!=0 and iteration!=0 and  iteration%intermediate_sampling==0:

                with open("save_state.pkl",'wb') as f:
                    pickle.dump([next_state,begin_index],f)
                saver=tf.train.Saver(tf.global_variables())
                saver.save(sess,"shakespeare_checkpoint/check",global_step=model.global_step)
                sample(iteration=iteration,checkpoint_dir="shakespeare_checkpoint/check-"+str(sess.run(model.global_step)))


            iteration = iteration + 1


        saver=tf.train.Saver(tf.global_variables())
        saver.save(sess,"shakespeare_checkpoint/final_check")








def main():
    train(intermediate_sampling=2500)
    #sample()
    #resume_training()

if __name__ == '__main__':
    main()





