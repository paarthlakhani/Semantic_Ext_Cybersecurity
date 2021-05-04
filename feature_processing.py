import data_processing
import numpy as np


class FeatureProcessing:
    def __init__(self):
        self.dp = data_processing.DataProcessing()

        # word, POS, label
        #self.train_table = np.empty([0, 3])
        self.train_table = np.empty([0, 1])
        self.test_table = np.empty([0, 1])
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
                #self.train_table = np.array(tokens)
                current_story_tokens = np.array(tokens)
                current_story_tokens = np.expand_dims(current_story_tokens, 1)
                self.train_table = np.concatenate((self.train_table, current_story_tokens), axis=0)
            else:
                current_story_tokens = np.array(tokens)
                current_story_tokens = np.expand_dims(current_story_tokens, 1)
                self.test_table = np.concatenate((self.test_table, current_story_tokens), axis=0)

    # sentences_of_story
    def sentences_of_story(self, tokenized_table, train = 1):
        sentences = []
        label_sentences = []

        sentence = ""
        label_sentence = ""

        for word_attr in tokenized_table:
            if word_attr != '\n':
                word_attr_arr = word_attr[0].split()
                sentence = sentence + word_attr_arr[0] + " "
                label_sentence = label_sentence + word_attr_arr[2] + " "
            else:
                sentence = sentence.strip()
                label_sentence = label_sentence.strip()
                sentences.append(sentence)
                label_sentences.append(label_sentence)
                sentence = ""
                label_sentence = ""

        return sentences, label_sentences


    def create_test_train_token_tables(self):
        self.dp.data_processing_main()
        self.read_token_files_into_tables(featureP.dp.train_token_file_contents, 1)
        self.read_token_files_into_tables(featureP.dp.test_token_file_contents, 0)


featureP = FeatureProcessing()
featureP.dp.data_processing_main()
featureP.read_token_files_into_tables(featureP.dp.train_token_file_contents, 1)
featureP.read_token_files_into_tables(featureP.dp.test_token_file_contents, 0)
train_sentences, train_label_sentences = featureP.sentences_of_story(featureP.train_table)
test_sentences, test_label_sentences = featureP.sentences_of_story(featureP.test_table)