import re
from utils.file_ops import read_file


def pre_process(raw_sentence):
    """
    :argument:
        :param raw_sentence: String
    :returns:
        sentence: String having only alphanumeric characters, space/s and Set("'", ",", "!", "?", ".")
    """
    sentence = re.sub(r"[^a-zA-Z0-9\s\'?!,.]", r"", raw_sentence)
    return sentence


def remove_extra_spaces(raw_sentence):
    """
    :argument:
        :param raw_sentence: String
    :returns:
        clean_str: String which does not contain new line character and more than 1 continuous white spaces.
    """
    clean_str = re.sub(r"\s\s+|\n", r" ", raw_sentence)
    return clean_str


def read_raw_data(file_name):
    """
    :argument:
        :param file_name: A string which represents the raw data file, in properties.conf,
                          used for the process (experiment).
    :Expects:
        Expected line format "coarse_class:fine_class This is the question string"
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
    flag, file = read_file(file_name)
    if flag:
        for line in file:
            space_separated_row = line.split(" ")
            classes = space_separated_row[0].split(":")
            question = " ".join(space_separated_row[1:])
            coarse_class, fine_class = classes[0], classes[1]
            coarse_classes_list.append(coarse_class)
            fine_classes_list.append(fine_class)
            questions_list.append(question)
        file.close()
        return True, coarse_classes_list, fine_classes_list, questions_list
    else:
        return False


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
