from os import listdir
from os import path
import xml.etree.ElementTree as et
import numpy as np
from io import StringIO


def sequence_label_dict():
    seq_label = {
        "O": 0,
        "B-Action": 1,
        "I-Action": 2,
        "B-Entity": 3,
        "I-Entity": 4,
        "B-Modifier": 5,
        "I-Modifier": 6
    }
    return seq_label


class DataProcessing:
    def __init__(self):
        self.dataset_path = './MalwareTextDB-1/data'
        self.plaintext_path = self.dataset_path + "/plaintext/"
        self.annotations_path = self.dataset_path + "/annotations/"
        self.tokenized_path = self.dataset_path + "/tokenized/"

        self.train_token_file_contents = None
        self.test_token_file_contents = None
        self.seq_label = sequence_label_dict()

    def read_ann_txt_filenames(self):
        all_fns = listdir(self.annotations_path)
        fn = []
        for filename in all_fns:
            if filename.endswith('.txt'):
                fn.append(filename)
        return fn

    def read_plaintext_filenames(self):
        all_fns = listdir(self.plaintext_path)
        return all_fns

    def get_filename_list(self, start, end, entire_filename_list):
        altered_filename_list = []
        for index in range(start, end):
            filename = entire_filename_list[index]
            altered_filename_list.append(filename)
        return altered_filename_list

    def read_plaintext_file_contents(self, filenames):
        filename_content_dict = {}
        for filename in filenames:
            filepath = self.plaintext_path + filename
            with open(filepath) as file_driver:
                file_content = file_driver.readlines()
                filename_content_dict[filename] = file_content
        return filename_content_dict

    def read_tokenized_file_contents(self, filenames):
        filename_token_content_dict = {}
        for file in filenames:
            # pre, ext = path.splitext(file)
            fileparts = file.split('.')
            filename = '.'.join(fileparts[:-2])
            token_filename = filename + '.tokens'
            token_fpath = self.tokenized_path + token_filename

            with open(token_fpath) as file_content:
                token_content = file_content.readlines()
                #token_content = file_content.read()
                filename_token_content_dict[filename] = token_content

        return filename_token_content_dict

    def read_ann_xml_files(self, filenames):
        ''' filename = 'Compromise_Greece_Beijing.txt'
        filepath = annotations_path + filename
        DOMTree = xml.dom.minidom.parse(filepath, encoding='utf-8')
        print(DOMTree)'''

        parser = et.XMLParser(encoding="UTF-8")
        for filename in filenames:
            # filename = 'Compromise_Greece_Beijing.xml'
            filename = 'Compromise_Greece_Beijing_2.txt'
            filepath = self.annotations_path + filename
            tree = et.parse(filepath, parser=parser)
            print(tree)

            # with open(filepath) as file_driver:
            #    file_content = file_driver.read()
            #    print(type(file_content))
            #    #file_content_lines = file_driver.readlines()
            #    tree = et.parse(file_content, parser=parser)
            #   print(tree)

    def data_processing_main(self):
        filenames = self.read_plaintext_filenames()
        number_filenames = len(filenames)
        train_number = 25 # 25
        test_number = 14 # 14
        eval_number = number_filenames - train_number - test_number
        # get train filename list
        train_filenames = self.get_filename_list(0, train_number, filenames)
        self.train_token_file_contents = self.read_tokenized_file_contents(train_filenames)

        # get test filename list
        test_filenames = self.get_filename_list(train_number, train_number + test_number, filenames)
        self.test_token_file_contents = self.read_tokenized_file_contents(test_filenames)
        # test_file_content = read_files(test_filenames)


if __name__ == '__main__':
    dp = DataProcessing()
    dp.data_processing_main()

'''
Complete by tomorrow
Things to do:
1. read xml files and read the lines before the footer
2. split each line from one word into separate line and every line should be separated by an empty line
3. Save the above files in a folder called evaluations or something
4. Write the evaluation script that compared this generated file with the one already created

For training:
1. Read the tokens file, combine every line along with the tags and BIO tagging, don't consider the txt files
2. Do the training with CRFs
3. combine the train sentences
'''
