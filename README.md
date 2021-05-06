This project focuses on SemEval 2018 Task 8
https://competitions.codalab.org/competitions/17262 

This SemEval task consists of 4 subtasks of which our system focuses on the second subtask:
Given a set of sentences, our system predicts the token labels. The token labels are: Entity, Action and Modifier and they are in BIO format.


We have explored two important ideas and algorithms to implement this subtask:
1. **Conditional Random Fields (CRF)**
CRFs was discussed in class that helped me decide this approach. These are Markov models that provide added advantage over HMMs i.e. it allows features to be used in addition to looking at the past history. So, it provides much more robust and better results. Following features are used: current word, previous word, previous two words, next word, next two words, POS tag for current/previous/previous two/next/next two words, whether the word is capitalized, whether the word is in the title format, whether the word is a digit, totaling to at most 33 features.

2. **Bi Long short-term memory (BiLSTM)**
Long short-term is an artificial recurrent neural network (RNN) architecture used in deep learning. LSTMs processes entire sequences of data and then produces the output, thus taking advantage of entire context rather than just few words. LSTMs process input once from left to right. BiLSTM, on the other hand, processes the input twice: once in forward direction from left
to right and then again in backward direction from right to left. A noted advantage of LSTM over CRF is of relative insensitivity thus, they can detect relations that are separated far away. As part of the implementation, we explored the ideas of building recurrent neural networks, loss functions, and backward propagation. Another important idea that was explored is of word embeddings using GloVe. Our input goes through word embeddings layer and then it goes to the LSTM model.

For Word Embeddings, we have used pretrained word vectors from the GloVE: https://nlp.stanford.edu/projects/glove/. More specifically, used the Wikipedia 2014 + Gigaword 5: glove.6B.zip.
Due to size constraints, I couldn't upload them.

To run the CRF implementation, run the crf_training.py
To run the BiLSTM implementation, run bilstm_main.py 