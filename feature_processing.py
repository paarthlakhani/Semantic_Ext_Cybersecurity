import data_processing
import numpy as np


class FeatureProcessing:
    def __init__(self):
        self.dp = data_processing.DataProcessing()

        # word, POS, label
        #self.train_table = np.empty([0, 3])
        self.train_table = np.empty([0, 3])
        self.test_table = np.empty([0, 3])
        #self.train_sentences = []  # This will all sentences for all stories

    def is_POS(self, potential_pos):
        if potential_pos.isalpha():
            return True
        return False

    '''def read_token_files_into_tables(self, token_file_contents, train = 1):
        # print(self.dp.data_processing_main())
        #train_token_file_contents = self.dp.train_token_file_contents

        for filename, tokens in token_file_contents.items():
            np_arr = np.array(tokens)
            np_arr = np.expand_dims(np_arr, 1)
            #sentence = ''
            for idx, token in enumerate(tokens):
                if token != '\n':
                    word_attrs = token.split()
                    #sentence = sentence + ' ' + word_attrs[0]
                    word_features = np.array(word_attrs)
                    if not self.is_POS(word_features[1]):
                        word_features[1] = 'NA'
                    word_features = word_features.reshape(1, 3)
                    if train:
                        self.train_table = np.append(self.train_table, word_features, axis=0)
                    else:
                        self.test_table = np.append(self.test_table, word_features, axis=0)
                else:
                    new_line_word = np.zeros([1, 3])
                    #sentence = sentence.strip()
                    #self.train_sentences.append(sentence)
                    if train:
                        self.train_table = np.append(self.train_table, new_line_word, axis=0)
                    else:
                        self.test_table = np.append(self.test_table, new_line_word, axis=0)
                    #sentence = '''''

    def read_token_files_into_tables(self, token_file_contents, train = 1):
        for filename, tokens in token_file_contents.items():
            if train:
                self.train_table = np.array(tokens)
                self.train_table = np.expand_dims(self.train_table, 1)
            else:
                self.test_table = np.array(tokens)
                self.test_table = np.expand_dims(self.test_table, 1)

    def create_test_train_token_tables(self):
        self.dp.data_processing_main()
        self.read_token_files_into_tables(featureP.dp.train_token_file_contents, 1)
        self.read_token_files_into_tables(featureP.dp.test_token_file_contents, 0)


featureP = FeatureProcessing()
featureP.dp.data_processing_main()
featureP.read_token_files_into_tables(featureP.dp.train_token_file_contents, 1)
featureP.read_token_files_into_tables(featureP.dp.test_token_file_contents, 0)
