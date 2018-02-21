from qc.utils.file_ops import write_str_file
from qc.utils.file_ops import read_file
from multiprocessing.pool import ThreadPool
import re


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
        clean_str: String which does not contain more than 1 continuous white spaces.
    """
    clean_str = re.sub(r"\s\s+", r" ", raw_sentence)
    return clean_str


def remove_space_before_apost(raw_sentence):
    """
    :argument:
        :param raw_sentence: String
    :returns:
        clean_str: String which does not contain more than 1 continuous white spaces.
    """
    clean_str = re.sub(r"\s'", r"'", raw_sentence)
    return clean_str


def remove_endline_char(raw_sentence):
    """
    :argument:
        :param raw_sentence: String
    :returns:
        clean_str: String which does not contain new line character
    """
    clean_str = re.sub(r"\n", r"", raw_sentence)
    return clean_str


def read_raw_data(file_key, rp):
    """
    :argument:
        :param file_key: A string which represents the raw data file, in properties.conf,
                          used for the process (experiment).
        :param rp: Absolute path of the root directory of the project

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
    flag, file = read_file(file_key, rp)
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
        c = remove_space_before_apost(remove_extra_spaces(pre_process(q)))
        n = remove_endline_char(c)
        clean_questions_list.append(n)
    return clean_questions_list


def dataset_raw_prep(data_type, rp: str):
    """
    :argument:
        :param data_type: String either `training` or `test`
        :param rp: Absolute path of the root directory of the project
    :Execution:
        Calls various raw_processing functions on the given raw text data
    :return:
        boolean flag: True for successful operation
    """
    data = "training" if data_type == "training" else "test"
    flag, coarse_class, fine_class, questions = read_raw_data("{0}_data".format(data), rp)
    if flag:
        q_clean = clean_sentences(questions)
        c = write_str_file(coarse_class, "coarse_classes_{0}".format(data), rp)
        f = write_str_file(fine_class, "fine_classes_{0}".format(data), rp)
        q = write_str_file(q_clean, "raw_sentence_{0}".format(data), rp)
        if not q:
            print("- Error while writing questions file for " + data)
            return False
        if not c:
            print("- Error while writing coarse class file for " + data)
            return False
        if not f:
            print("- Error while writing fine class file for " + data)
            return False
        return True
    else:
        print("- Error is reading and splitting " + data + " data")
        return False


def execute(project_root_path: str):
    print("\n* Raw Data Processing")
    # Create two threads for processing training and test raw files
    pool = ThreadPool(processes=2)
    # start the threads and wait for them to finish
    train_result = pool.apply_async(dataset_raw_prep, args=["training", project_root_path])
    test_result = pool.apply_async(dataset_raw_prep, args=["test", project_root_path])
    train_val = train_result.get()
    test_val = test_result.get()
    if not train_val:
        print("- Error: In text splitting for training data")
    if not test_val:
        print("- Error: In text splitting for test data")
    if train_val and test_val:
        print("- Raw text splitting done for training and test data")