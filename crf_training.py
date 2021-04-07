import feature_processing
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from collections import Counter


class CRFTraining:
    def __init__(self):
        self.featureP = feature_processing.FeatureProcessing()
        self.featureP.create_train_table()
        self.word_attribute_table = self.featureP.train_table
        self.train_sentences_list = self.featureP.train_sentences

    def word2features(self, i):
        # Extract features from every word and convert word to a feature vector for CRF model
        word = self.word_attribute_table[i][1]
        pos_tag = self.word_attribute_table[i][2]
        token_label = self.word_attribute_table[i][3]

        features = {
            'word.lower()': word.lower()
        }

        return features
