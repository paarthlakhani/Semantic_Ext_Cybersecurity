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
            'postag[:2]': pos_tag[:2],
            'word.lower()': word.lower(),
            'word': word,
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit()
        }

        if i != total_length - 1:
            if i >= 1:
                if self.train_table_word_attrs[i - 1] != '\n':
                    previous_word_attr = self.train_table_word_attrs[i - 1][0].split()
                    word_before = previous_word_attr[0]
                    features['word_unigram_before'] = word_before
                    features['-1:word.lower()'] = word_before.lower()
                    features['-1:word.istitle()'] = word_before.istitle()
                    features['-1:word.isupper()'] = word_before.isupper()
                    features['pos_unigram_before'] = previous_word_attr[1]
                    features['-1:postag[:2]']: previous_word_attr[1][:2]
            if i >= 2:
                if self.train_table_word_attrs[i - 1] != '\n' and self.train_table_word_attrs[i - 2] != '\n':
                    previous_word_attr_1 = self.train_table_word_attrs[i - 1][0].split()
                    previous_word_attr_2 = self.train_table_word_attrs[i - 2][0].split()
                    word_before_1 = previous_word_attr_1[0]
                    word_before_2 = previous_word_attr_2[0]
                    bigram_before = word_before_1 + " " + word_before_2
                    #bigram_before = word_before_2
                    features['word_bigram_before'] = bigram_before
                    features['-2:word.lower()'] = bigram_before.lower()
                    features['-2:word.istitle()'] = bigram_before.istitle()
                    features['-2:word.isupper()'] = bigram_before.isupper()
                    features['pos_bigram_before'] = previous_word_attr_1[1] + " " + previous_word_attr_2[1]
                    #features['pos_bigram_before'] = previous_word_attr_2[1]
                    #features['-2:postag[:2]']: previous_word_attr_2[1][:2]
                    features['-2:postag[:2]']: previous_word_attr_1[1][:2] + " " + previous_word_attr_2[1][:2]
            if i <= total_length - 2:
                if self.train_table_word_attrs[i + 1] != '\n':
                    next_word_attr = self.train_table_word_attrs[i + 1][0].split()
                    word_after = next_word_attr[0]
                    features['word_unigram_after'] = word_after
                    features['+1:word.lower()'] = word_after.lower()
                    features['+1:word.istitle()'] = word_after.istitle()
                    features['+1:word.isupper()'] = word_after.isupper()
                    features['pos_unigram_after'] = next_word_attr[1]
                    features['+1:postag[:2]']: next_word_attr[1][:2]
            if i <= total_length - 3:
                if self.train_table_word_attrs[i + 1] != '\n' and self.train_table_word_attrs[i + 2] != '\n':
                    next_word_attr_1 = self.train_table_word_attrs[i + 1][0].split()
                    next_word_attr_2 = self.train_table_word_attrs[i + 2][0].split()
                    word_after_1 = next_word_attr_1[0]
                    word_after_2 = next_word_attr_2[0]
                    bigram_after = word_after_1 + " " + word_after_2
                    #bigram_after = word_after_2
                    features['word_bigram_after'] = bigram_after
                    features['+2:word.lower()'] = bigram_after.lower()
                    features['+2:word.istitle()'] = bigram_after.istitle()
                    features['+2:word.isupper()'] = bigram_after.isupper()
                    features['pos_bigram_after'] = next_word_attr_1[1] + " " + next_word_attr_2[1]
                    #features['pos_bigram_after'] = next_word_attr_2[1]
                    #features['+2:postag[:2]']: next_word_attr_2[1][:2]
                    features['+2:postag[:2]']: next_word_attr_1[1][:2] + " " + next_word_attr_2[1][:2]
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
            'postag[:2]': pos_tag[:2],
            'word.lower()': word.lower(),
            'word': word,
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit()
        }

        if i != total_length - 1:
            if i >= 1:
                if self.test_table_word_attrs[i - 1] != '\n':
                    previous_word_attr = self.test_table_word_attrs[i - 1][0].split()
                    word_before = previous_word_attr[0]
                    features['word_unigram_before'] = word_before
                    features['-1:word.lower()'] = word_before.lower()
                    features['-1:word.istitle()'] = word_before.istitle()
                    features['-1:word.isupper()'] = word_before.isupper()
                    features['pos_unigram_before'] = previous_word_attr[1]
                    features['-1:postag[:2]']: previous_word_attr[1][:2]
            if i >= 2:
                if self.test_table_word_attrs[i - 1] != '\n' and self.test_table_word_attrs[i - 2] != '\n':
                    previous_word_attr_1 = self.test_table_word_attrs[i - 1][0].split()
                    previous_word_attr_2 = self.test_table_word_attrs[i - 2][0].split()
                    word_before_1 = previous_word_attr_1[0]
                    word_before_2 = previous_word_attr_2[0]
                    bigram_before = word_before_1 + " " + word_before_2
                    #bigram_before = word_before_2
                    features['word_bigram_before'] = bigram_before
                    features['-2:word.lower()'] = bigram_before.lower()
                    features['-2:word.istitle()'] = bigram_before.istitle()
                    features['-2:word.isupper()'] = bigram_before.isupper()
                    features['pos_bigram_before'] = previous_word_attr_1[1] + " " + previous_word_attr_2[1]
                    #features['pos_bigram_before'] = previous_word_attr_2[1]
                    #features['-2:postag[:2]']: previous_word_attr_2[1][:2]
                    features['-2:postag[:2]']: previous_word_attr_1[1][:2] + " " + previous_word_attr_2[1][:2]
            if i <= total_length - 2:
                if self.test_table_word_attrs[i + 1] != '\n':
                    next_word_attr = self.test_table_word_attrs[i + 1][0].split()
                    word_after = next_word_attr[0]
                    features['word_unigram_after'] = word_after
                    features['+1:word.lower()'] = word_after.lower()
                    features['+1:word.istitle()'] = word_after.istitle()
                    features['+1:word.isupper()'] = word_after.isupper()
                    features['pos_unigram_after'] = next_word_attr[1]
                    features['+1:postag[:2]']: next_word_attr[1][:2]
            if i <= total_length - 3:
                if self.test_table_word_attrs[i + 1] != '\n' and self.test_table_word_attrs[i + 2] != '\n':
                    next_word_attr_1 = self.test_table_word_attrs[i + 1][0].split()
                    next_word_attr_2 = self.test_table_word_attrs[i + 2][0].split()
                    word_after_1 = next_word_attr_1[0]
                    word_after_2 = next_word_attr_2[0]
                    bigram_after = word_after_1 + " " + word_after_2
                    #bigram_after = word_after_2
                    features['word_bigram_after'] = bigram_after
                    features['+2:word.lower()'] = bigram_after.lower()
                    features['+2:word.istitle()'] = bigram_after.istitle()
                    features['+2:word.isupper()'] = bigram_after.isupper()
                    features['pos_bigram_after'] = next_word_attr_1[1] + " " + next_word_attr_2[1]
                    #features['pos_bigram_after'] = next_word_attr_2[1]
                    #features['+2:postag[:2]']: next_word_attr_2[1][:2]
                    features['+2:postag[:2]']: next_word_attr_1[1][:2] + " " + next_word_attr_2[1][:2]
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
            max_iterations=1000,
            all_possible_transitions=True
        )

        self.crf_model.fit(x_train, y_train)

    def test_crf(self):
        x_test = [self.create_x_test()]
        y_test = [self.create_y_label_for_test()]

        labels = list(self.crf_model.classes_)
        #labels.remove('O')
        y_pred = self.crf_model.predict(x_test)
        print(metrics.flat_f1_score(y_test, y_pred, average='weighted', labels = labels))

        sorted_labels = sorted(
            labels,
            key=lambda name: (name[1:], name[0])
        )
        print(metrics.flat_classification_report(
            y_test, y_pred, labels=sorted_labels, digits=3
        ))

        #metrics.flat_f1_score(y_test, y_pred, average='weighted', labels = labels)


crf_obj = CRFTraining()
crf_obj.train_crf()
crf_obj.test_crf()
