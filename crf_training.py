import feature_processing
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from collections import Counter


class CRFTraining:
    def __init__(self):
        self.featureP = feature_processing.FeatureProcessing()
        self.featureP.create_test_train_token_tables()
        #self.featureP.read_token_files_into_tables()
        self.train_table_word_attrs = self.featureP.train_table
        self.test_table_word_attrs = self.featureP.test_table
        #self.train_sentences_list = self.featureP.train_sentences  # Not needed
        self.crf_model = None


    def word2features_train(self, i):
        # Extract features from every word and convert word to a feature vector for CRF model
        total_length = self.train_table_word_attrs.shape[0]
        word_attrs = self.train_table_word_attrs[i][0].split()
        word = word_attrs[0]
        pos_tag = word_attrs[1]
        # token_label = self.word_attribute_table[i][2]

        features = {
            'postag': pos_tag,
            'word.lower()': word.lower()
        }

        if i != total_length - 1:
            if i >= 1:
                if self.train_table_word_attrs[i - 1] != '\n':
                    word_before = self.train_table_word_attrs[i - 1][0].split()[0]
                    features['unigram_before'] = word_before
            if i >= 2:
                if self.train_table_word_attrs[i - 1] != '\n' and self.train_table_word_attrs[i - 2] != '\n':
                    word_before_1 = self.train_table_word_attrs[i - 1][0].split()[0]
                    word_before_2 = self.train_table_word_attrs[i - 2][0].split()[0]
                    bigram_before = word_before_1 + " " + word_before_2
                    features['bigram_before'] = bigram_before
            if i <= total_length - 2:
                if self.train_table_word_attrs[i + 1] != '\n':
                    word_after = self.train_table_word_attrs[i + 1][0].split()[0]
                    features['unigram_after'] = word_after
            if i <= total_length - 3:
                if self.train_table_word_attrs[i + 1] != '\n' and self.train_table_word_attrs[i + 2] != '\n':
                    word_after_1 = self.train_table_word_attrs[i + 1][0].split()[0]
                    word_after_2 = self.train_table_word_attrs[i + 2][0].split()[0]
                    bigram_after = word_after_1 + " " + word_after_2
                    features['bigram_after'] = bigram_after
        return features

    def word2features_test(self, i):
        # Extract features from every word and convert word to a feature vector for CRF model
        total_length = self.test_table_word_attrs.shape[0]
        word_attrs = self.test_table_word_attrs[i][0].split()
        word = word_attrs[0]
        pos_tag = word_attrs[1]
        # token_label = self.word_attribute_table[i][2]

        features = {
            'postag': pos_tag,
            'word.lower()': word.lower()
        }

        if i != total_length - 1:
            if i >= 1:
                if self.test_table_word_attrs[i - 1] != '\n':
                    word_before = self.test_table_word_attrs[i - 1][0].split()[0]
                    features['unigram_before'] = word_before
            if i >= 2:
                if self.test_table_word_attrs[i - 1] != '\n' and self.test_table_word_attrs[i - 2] != '\n':
                    word_before_1 = self.test_table_word_attrs[i - 1][0].split()[0]
                    word_before_2 = self.test_table_word_attrs[i - 2][0].split()[0]
                    bigram_before = word_before_1 + " " + word_before_2
                    features['bigram_before'] = bigram_before
            if i <= total_length - 2:
                if self.test_table_word_attrs[i + 1] != '\n':
                    word_after = self.test_table_word_attrs[i + 1][0].split()[0]
                    features['unigram_after'] = word_after
            if i <= total_length - 3:
                if self.test_table_word_attrs[i + 1] != '\n' and self.test_table_word_attrs[i + 2] != '\n':
                    word_after_1 = self.test_table_word_attrs[i + 1][0].split()[0]
                    word_after_2 = self.test_table_word_attrs[i + 2][0].split()[0]
                    bigram_after = word_after_1 + " " + word_after_2
                    features['bigram_after'] = bigram_after

        return features

    def create_x_train(self):
        x_train = []
        #for i, word_attr in enumerate(self.train_table_word_attrs):
        for i in range(self.train_table_word_attrs.shape[0]):
            if self.train_table_word_attrs[i] != '\n':
                x_train.append(self.word2features_train(i))
        return x_train

    def create_y_label_for_train(self):
        y_label = []
        for i, word_attr in enumerate(self.train_table_word_attrs):
            if word_attr != '\n':
                y_label.append(word_attr[0].split()[2])
        return y_label

    def create_x_test(self):
        x_test = []
        for i in range(self.test_table_word_attrs.shape[0]):
            if self.test_table_word_attrs[i] != '\n':
                x_test.append(self.word2features_test(i))
        return x_test

    def create_y_label_for_test(self):
        y_label_test = []
        for i, word_attr in enumerate(self.test_table_word_attrs):
            if word_attr != '\n':
                y_label_test.append(word_attr[0].split()[2])
        return y_label_test

    def train_crf(self):
        x_train = [self.create_x_train()]
        y_train = [self.create_y_label_for_train()]

        self.crf_model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )

        self.crf_model.fit(x_train, y_train)

    def test_crf(self):
        x_test = [self.create_x_test()]
        y_test = [self.create_y_label_for_test()]

        labels = list(self.crf_model.classes_)
        labels.remove('O')
        y_pred = self.crf_model.predict(x_test)
        print(metrics.flat_f1_score(y_test, y_pred, average='weighted', labels = labels))

        #metrics.flat_f1_score(y_test, y_pred, average='weighted', labels = labels)


crf_obj = CRFTraining()
crf_obj.train_crf()
crf_obj.test_crf()
