from bilstm_model import BiLSTMModel
from data_processing import sequence_label_dict
import feature_processing
import word_embeddings as embed
import torch.nn as nn
import numpy as np

import torch


class BiLSTMMain:
    def __init__(self):
        self.featureP = feature_processing.FeatureProcessing()
        self.featureP.create_test_train_token_tables()
        self.train_sentences, self.train_label_sentences = self.featureP.sentences_of_story(self.featureP.train_table)
        self.test_sentences, self.test_label_sentences = self.featureP.sentences_of_story(self.featureP.test_table)
        self.bilstm_model = None
        self.embedding_layer = None
        self.eval_dict = {}

    def create_model(self, input_size, hidden_size):
        self.bilstm_model = BiLSTMModel(input_size, hidden_size)  # 300, 100

    def get_optmizer(self, learning_rate, weight_decay):
        optimizer = torch.optim.Adam(self.bilstm_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        return optimizer

    def compute_loss(self, predicted_output_data, true_label_for_sentence):
        loss = self.bilstm_model.loss_criterion(predicted_output_data, true_label_for_sentence)
        return loss

    def label_for_sentence_in_tensor(self, true_label_for_sentence_string):
        sequence_labels_dict = sequence_label_dict()
        true_labels = true_label_for_sentence_string.split()
        true_labels_arr = []
        for true_label in true_labels:
            true_label_mapping = sequence_labels_dict[true_label]
            true_labels_arr.append(true_label_mapping)
        return torch.tensor(true_labels_arr)

    def predict_labels(self, predicted_output_probs_of_labels):
        predicted_labels = torch.argmax(predicted_output_probs_of_labels, dim=1)
        return predicted_labels

    def train_model(self, num_epochs=10):
        optimizer = self.get_optmizer(learning_rate=0.0001, weight_decay=0.0001)
        self.bilstm_model.train()
        #print("Number of sentences: " + str(len(self.train_sentences)))
        for i in range(num_epochs):
            tot_loss = 0
            for idx, sentence in enumerate(self.train_sentences):
                #print("Sentence number: " + str(idx))
                sentence_words = sentence.split()
                sentence_words_idxs = [word_embed.get_embedding(word) for word in sentence_words]
                input_x = (self.embedding_layer.weight[sentence_words_idxs[0]]).view(1, -1)
                for sentence_words_idx in range(1, len(sentence_words_idxs)):
                    current_word_embedding = (self.embedding_layer.weight[sentence_words_idxs[sentence_words_idx]]).view(1, -1)
                    input_x = torch.cat((input_x, current_word_embedding), dim=0)
                true_label_for_sentence_in_tensor = self.label_for_sentence_in_tensor(self.train_label_sentences[idx])
                optimizer.zero_grad()
                input_x = torch.unsqueeze(input_x, 0)
                predicted_output_probs_of_labels = self.bilstm_model(input_x)
                predicted_output_probs_of_labels = torch.squeeze(predicted_output_probs_of_labels, 0)
                #predicted_output_labels = self.predict_labels(predicted_output_probs_of_labels)
                loss = self.compute_loss(predicted_output_probs_of_labels, true_label_for_sentence_in_tensor)

                individual_loss = [0, 0, 0, 0, 0, 0, 0]
                for (j, o) in enumerate(true_label_for_sentence_in_tensor):  # entire_y[i]
                    individual_loss[true_label_for_sentence_in_tensor[j]] += loss[j]
                # print(loss)
                for j in range(7):
                    #if ((torch.tensor(true_label_for_sentence_in_tensor) == j).sum().item() > 0.1):
                    if ((true_label_for_sentence_in_tensor.clone().detach() == j).sum().item() > 0.1):
                        #individual_loss[j] /= ((torch.tensor(true_label_for_sentence_in_tensor) == j).sum().item())
                        individual_loss[j] /= ((true_label_for_sentence_in_tensor.clone().detach() == j).sum().item())
                # print(loss)
                loss_final = 0
                for lo in individual_loss:
                    loss_final += lo
                tot_loss += loss_final

                loss_final.backward()   # loss.backward()
                #loss.backward()
                optimizer.step()
            print("Epoch number:{}, Loss:{:.4f}".format(i + 1, float(tot_loss)))  # tot_loss if using above

        torch.save(self.bilstm_model, "bilstm_model.pt")

    def test_model(self):
        all_true_labels = []
        all_predicted_labels = []
        if self.bilstm_model is None:
            self.bilstm_model = torch.load("bilstm_model.pt")
        #self.bilstm_model.eval()
        with torch.no_grad():
            for idx, sentence in enumerate(self.test_sentences):
                sentence_words = sentence.split()
                sentence_words_idxs = [word_embed.get_embedding(word) for word in sentence_words]
                input_x = (self.embedding_layer.weight[sentence_words_idxs[0]]).view(1, -1)
                for sentence_words_idx in range(1, len(sentence_words_idxs)):
                    current_word_embedding = (self.embedding_layer.weight[sentence_words_idxs[sentence_words_idx]]).view(1, -1)
                    input_x = torch.cat((input_x, current_word_embedding), dim=0)
                true_label_for_sentence_in_tensor = self.label_for_sentence_in_tensor(self.test_label_sentences[idx])
                input_x = torch.unsqueeze(input_x, 0)
                predicted_output_probs_of_labels = self.bilstm_model(input_x)
                predicted_output_probs_of_labels = torch.squeeze(predicted_output_probs_of_labels, 0)
                predicted_output_labels = self.predict_labels(predicted_output_probs_of_labels)
                all_true_labels.extend(true_label_for_sentence_in_tensor)
                all_predicted_labels.extend(predicted_output_labels)
            self.evaluate_sentence_metrics(torch.tensor(all_true_labels), torch.tensor(all_predicted_labels))
                #print("True labels:")
                #print(true_label_for_sentence_in_tensor)
                #print("Predicted labels:")
                #print(predicted_output_labels)

    def evaluate_sentence_metrics(self, all_true_labels, all_predicted_labels):
        #print(all_true_labels)
        #print(all_predicted_labels)
        recall = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        precision = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        f_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        different_labels, all_true_label_counts = torch.unique(all_true_labels, return_counts=True)
        different_labels, all_predicted_label_counts = torch.unique(all_predicted_labels, return_counts=True)
        correctly_labeled = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for idx, true_label in enumerate(all_true_labels):
            if true_label == all_predicted_labels[idx]:
                correctly_labeled[true_label] = correctly_labeled[true_label] + 1
        for i in range(7):
            recall[i] = float(correctly_labeled[i]/all_true_label_counts[i])
            precision[i] = float(correctly_labeled[i]/all_predicted_label_counts[i])
            f_score[i] = float(2*recall[i]*precision[i]/(recall[i] + precision[i]))
        print(recall)
        print(precision)
        print(f_score)
        f_score_tensor = torch.tensor(f_score, dtype=torch.float32)
        f_score_mean = torch.mean(f_score_tensor)
        print(f_score_mean)


bi_main = BiLSTMMain()
word_embed = embed.WordEmbeddings()
word_to_idx, vectors = word_embed.read_word_embeddings()
bi_main.embedding_layer = nn.Embedding.from_pretrained(torch.from_numpy(vectors.astype(np.float32)))
#print(bi_main.embedding_layer.weight)
#print("Hello")
bi_main.create_model(300, 100)
bi_main.train_model()  # need to pass word_embed?
bi_main.test_model()
