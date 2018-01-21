import os
import re
from utils import read_operations


def pre_process(raw_sentence):
    """
    :argument:
        :param raw_sentence: String
    :returns:
        sentence: String having only alphanumeric characters and space/s
    """
    sentence = re.sub(r'[^a-zA-Z0-9\s]', r'', raw_sentence)
    return sentence


def remove_extra_spaces(raw_sentence):
    """
    :argument:
        :param raw_sentence: String
    :returns:
        clean_str: String which does not contain more than 1 continuous white spaces.
    """
    clean_str = re.sub(r'\s\s+', r' ', raw_sentence)
    return clean_str


def read_raw_data(file_name):
    """
    :argument:
        :param file_name: A string which represents the raw data file, in properties.conf,
                          used for the process (experiment).
        Note: File should be using UTF-8 encoding. Change the encoding as needed.
    :exception:
        :except IOError: This may occur because of many reasons. e.g file is missing or corrupt file or wrong file path
    :Expects:
        Expected line format "coarse_class::fine_class This is the question string"
    :returns:
        boolean flag: True for successful operation
        coarse_classes_list: List of coarse classes for questions
        fine_classes_list: List of fine classes for questions
        questions_list: List of the questions in the raw data file
    :Example for a single line:
        ENTY:cremat What films featured the character Popeye Doyle ?
        :return: coarse_classes_list = ['ENTY'], fine_classes_list = ['cremat'],
                 questions_list = ['What films featured the character Popeye Doyle ?']
    """
    coarse_classes_list = []
    fine_classes_list = []
    questions_list = []
    try:
        with open(read_operations.read_file(file_name), "r", encoding='utf8') as file:
            for line in file:
                space_separated_row = line.split(" ")
                classes = space_separated_row[0].split(":")
                question = " ".join(space_separated_row[1:])
                coarse_class, fine_class = classes[0], classes[1]
                coarse_classes_list.append(coarse_class)
                fine_classes_list.append(fine_class)
                questions_list.append(question)
    except IOError as e:
        print("File IO Error :: Cannot open " + read_operations.read_file(file_name) + "\n" + str(e))
        return False
    return True, coarse_classes_list, fine_classes_list, questions_list


def clean_sentences(questions_list):
    """
    :argument:
        :param questions_list: List of string
    :return:
        clean_questions_list: List of string containing only alphanumeric characters and non-continuous white spaces.
    """
    clean_questions_list = []
    for q in questions_list:
        c = remove_extra_spaces(pre_process(q))
        clean_questions_list.append(c)
    return clean_questions_list


def write_str_file(str_list, file_name):
    """
    :argument:
        :param str_list: List of string which are to written to the file with line endings
        :param file_name: String which represents the file from properties.conf
    :return:
        boolean flag: True for successful operation
    """
    if not os.path.exists(os.path.dirname(read_operations.read_file(file_name))):
        try:
            os.makedirs(os.path.dirname(read_operations.read_file(file_name)))
        except OSError as err:
            print("Error creating director " + read_operations.read_file(file_name) + "\n" + str(err))
            return False
    try:
        with open(read_operations.read_file(file_name), "w", encoding='utf8') as file:
            str_to_write = "\n".join(str_list)
            file.write(str_to_write)
    except IOError as e:
        print("File IO Error :: Cannot write to " + read_operations.read_file(file_name) + "\n" + str(e))
        return False
    return True
