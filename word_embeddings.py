import numpy as np


class WordEmbeddings:
    def __init__(self):
        self.word_to_idx = {}

    def get_idx_of_word(self, word):
        if word not in self.word_to_idx:
            return -1
        return self.word_to_idx[word]

    def get_embedding(self, word):
        word_idx = self.get_idx_of_word(word)
        if word_idx != -1:
            return word_idx
        else:
            return self.get_idx_of_word("UNK")

    def get_idx_word_or_add(self, word):
        if word not in self.word_to_idx:
            word_idx = len(self.word_to_idx)
            self.word_to_idx[word] = word_idx
        return self.word_to_idx[word]

    def read_word_embeddings(self, input_size=300):  # input_size = 50, 100, 200, 300
        directory_path = './glove.6B/glove.6B.' + str(input_size) + 'd.txt'
        f = open(directory_path)
        vectors = []
        for line in f:
            if line.strip() != "":
                space_idx = line.find(' ')
                word = line[:space_idx]
                numbers = line[space_idx + 1:]
                embeddings = [float(number_str) for number_str in numbers.split()]
                embeddings_vector = np.array(embeddings)
                self.get_idx_word_or_add(word)  # Adding into word -> idx dictionary
                vectors.append(embeddings_vector)
        f.close()
        return self.word_to_idx, np.array(vectors)

