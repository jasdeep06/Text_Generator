import numpy as np

class InputGenerator():

    def __init__(self,file_name,batch_size,sequence_length):
        self.filename=file_name
        self.batch_size=batch_size
        self.sequence_length=sequence_length

        f=open(self.filename)
        self.text=f.read()



    def get_vocab_size(self):
        unique_chars = []
        for character in self.text:
            if character not in unique_chars:
                unique_chars.append(character)
        vocab_size=len(unique_chars)
        return vocab_size,self.text




        """""

        characters = list(set(self.text))

        character_index_dict = {character: index for index, character in enumerate(characters)}
        index_character_dict = {index: character for index, character in enumerate(characters)}

        return vocab_size,character_index_dict,index_character_dict,self.text
        """""

    def char_to_index(self):
        value = -1
        char_dict = {}

        for character in self.text:
            if character in char_dict:
                continue
            else:
                value = value + 1
                char_dict[character] = value
        return char_dict

    def index_to_char(self,char_to_index):
        index_to_char = {y: x for x, y in char_to_index.items()}
        return index_to_char


    def generate_batches(self):


        begin_index = np.random.random_integers(len(self.text) - self.batch_size * self.sequence_length - 4)
        # print(begin_index)

        batch_text_x = self.text[begin_index:begin_index + self.batch_size * self.sequence_length]
        batch_text_y = self.text[begin_index + 1:begin_index + self.batch_size * self.sequence_length + 1]
        return batch_text_x, batch_text_y

    def process_batches(self,batch,char_to_index):

        targets = []
        for i in range(self.batch_size):
            internal_list = []

            for character in batch[i * self.sequence_length:self.sequence_length * (i + 1)]:
                internal_list.append(char_to_index[character])
            targets.append(internal_list)
        return targets


    def generate_ordered_batches(self,begin_index):

        batch_text_x=self.text[begin_index:begin_index+self.batch_size*self.sequence_length]
        batch_text_y = self.text[begin_index + 1:begin_index + self.batch_size * self.sequence_length + 1]

        return batch_text_x,batch_text_y

