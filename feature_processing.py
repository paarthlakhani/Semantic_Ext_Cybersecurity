import data_processing
import numpy as np


class FeatureProcessing:
    def __init__(self):
        self.dp = data_processing.DataProcessing()

        # word, POS, label
        self.train_table = np.empty([0, 3])
        self.train_sentences = []

    def is_POS(self, potential_pos):
        if potential_pos.isalpha():
            return True
        return False

    def create_train_table(self):
        # print(self.dp.data_processing_main())
        self.dp.data_processing_main()
        train_token_file_contents = self.dp.train_token_file_contents

        for filename, tokens in train_token_file_contents.items():
            sentence = ''
            for idx, token in enumerate(tokens):
                if token != '\n':
                    sentence = sentence + ' ' + token
                    word_features = np.array(token.split())
                    if not self.is_POS(word_features[1]):
                        word_features[1] = 'NA'
                    word_features = word_features.reshape(1, 3)
                    self.train_table = np.append(self.train_table, word_features, axis=0)
                else:
                    self.train_sentences.append(sentence)
                    sentence = ''


featureP = FeatureProcessing()
featureP.create_train_table()
