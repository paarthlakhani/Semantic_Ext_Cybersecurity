import data_processing
import numpy as np

class FeatureProcessing:
    def __init__(self):
        self.dp = data_processing.DataProcessing()

        # word, POS, label
        self.train_table = np.empty([0, 3])

    def create_train_table(self):
        # print(self.dp.data_processing_main())
        self.dp.data_processing_main()
        train_token_file_contents = self.dp.train_token_file_contents

        for filename, tokens in train_token_file_contents.items():
            for idx, token in enumerate(tokens):
                if token != '\n':
                    word_features = np.array(token.split())
                    word_features = word_features.reshape(1, 3)
                    self.train_table = np.append(self.train_table, word_features, axis=0)


featureP = FeatureProcessing()
featureP.create_train_table()
