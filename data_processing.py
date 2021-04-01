import numpy as np
from os import listdir

dataset_path = './MalwareTextDB-1/data'
annotations_path = dataset_path + "/plaintext/"


def read_filenames():
    fn = listdir(annotations_path)
    return fn


def get_filename_list(start, end, entire_filename_list):
    altered_filename_list = []
    for index in range(start, end):
        filename = entire_filename_list[index]
        altered_filename_list.append(filename)
    return altered_filename_list


def read_files(filenames):
    filename_content_dict = {}
    for filename in filenames:
        filepath = annotations_path + filename
        with open(filepath) as file_driver:
            file_content = file_driver.readlines()
            filename_content_dict[filename] = file_content
            # print("This is the file content")
            # print(file_content)
    return filename_content_dict


filenames = read_filenames()
number_filenames = len(filenames)
train_number = 25
test_number = 14
eval_number = number_filenames - train_number - test_number
# get train filename list
train_filenames = get_filename_list(0, train_number, filenames)
train_file_content = read_files(train_filenames)


# get test filename list
test_filenames = get_filename_list(train_number, train_number + test_number, filenames)
test_file_content = read_files(test_filenames)

