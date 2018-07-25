import re
from multiprocessing.pool import ThreadPool

from qc.utils.file_ops import read_file
from qc.utils.file_ops import write_str_file


def pre_process(raw_sentence):
    """
    This method removes the characters from sentence,
    which are not expected in grammatically correct questions.

    :argument:
        :param raw_sentence: String
    :returns:
        sentence: String having only alphanumeric characters, space/s and Set("'", ",", "!", "?", ".")
    """
    sentence = re.sub(r"[^a-zA-Z0-9\s\'?!,.]", r"", raw_sentence)
    return sentence


def remove_extra_spaces(raw_sentence):
    """
    This method removes more than one continuous spaces and reduces to only one space.

    :argument:
        :param raw_sentence: String
    :returns:
        clean_str: String which does not contain more than 1 continuous white spaces.
    """
    clean_str = re.sub(r"\s\s+", r" ", raw_sentence)
    return clean_str


def remove_space_before_apost(raw_sentence):
    """
    This method removes the space between a word and the apostrophe,
    to make it more correct grammatically.

    :argument:
        :param raw_sentence: String
    :returns:
        clean_str: String which does not contain more than 1 continuous white spaces.
    """
    clean_str = re.sub(r"\s'", r"'", raw_sentence)
    return clean_str


def remove_endline_char(raw_sentence):
    """
    This method removes all the end-line characters from a string.

    :argument:
        :param raw_sentence: String
    :returns:
        clean_str: String which does not contain new line character
    """
    clean_str = re.sub(r"\n", r"", raw_sentence)
    return clean_str


def read_raw_data(file_key, rp):
    """
    This method reads the dataset present as it is originally and
    forms the structure which will be easy to use for the further process.

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
    This method uses all the regex methods to clean the sentences of unwanted characters.

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


def sep_fine_classes(coarse, fine):
    """
    This method combines all the fine class of a particular coarse class together.

    :argument
        :param coarse: Coarse class of the given row
        :param fine: Fine class of the given row
    :return:
        abbr_class: List of class of questions belonging to ABBR coarse class.
        desc_class: List of class of questions belonging to DESC coarse class.
        enty_class: List of class questions belonging to ENTY coarse class.
        hum_class: List of class of questions belonging to HUM coarse class.
        loc_class: List of class of questions belonging to LOC coarse class.
        num_class: List of class of questions belonging to NUM coarse class.
    """
    abbr_class = []
    desc_class = []
    enty_class = []
    hum_class = []
    loc_class = []
    num_class = []
    i = 0
    for subclass in fine:
        if coarse[i] == "ABBR":
            abbr_class.append(subclass)
        elif coarse[i] == "DESC":
            desc_class.append(subclass)
        elif coarse[i] == "ENTY":
            enty_class.append(subclass)
        elif coarse[i] == "HUM":
            hum_class.append(subclass)
        elif coarse[i] == "LOC":
            loc_class.append(subclass)
        elif coarse[i] == "NUM":
            num_class.append(subclass)
        else:
            print("{0} is an unexpected coarse class".format(coarse[i]))
        # increment i by one, so that the proper lines are match
        i = i + 1
    return abbr_class, desc_class, enty_class, hum_class, loc_class, num_class


def dataset_raw_prep(data_type, rp: str):
    """
    This method handles the raw processing of dataset
    and process training or test data as per the arguments.

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
    if data == "training":
        abbr_class, desc_class, enty_class, hum_class, loc_class, num_class = sep_fine_classes(coarse_class, fine_class)
        a = write_str_file(abbr_class, "abbr_classes_{0}".format(data), rp)
        d = write_str_file(desc_class, "desc_classes_{0}".format(data), rp)
        e = write_str_file(enty_class, "enty_classes_{0}".format(data), rp)
        h = write_str_file(hum_class, "hum_classes_{0}".format(data), rp)
        lo = write_str_file(loc_class, "loc_classes_{0}".format(data), rp)
        n = write_str_file(num_class, "num_classes_{0}".format(data), rp)
        if not (a and d and e and h and lo and n):
            print("- Error while writing sub classes files for " + data)
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
    """
    This method starts 2 threads for processing the raw dataset.
    training|test

    :argument
        :param project_root_path: Absolute Path of the project
    :return:
        None
    """
    print("\n* Raw Data Processing")
    # Create two threads for processing training and test raw files
    pool = ThreadPool(processes=2)
    # start the threads and wait for them to finish
    train_result = pool.apply_async(dataset_raw_prep, args=["training", project_root_path])
    test_result = pool.apply_async(dataset_raw_prep, args=["test", project_root_path])
    train_status = train_result.get()
    test_status = test_result.get()
    if not train_status:
        print("- Error: In text splitting for training data")
    if not test_status:
        print("- Error: In text splitting for test data")
    if train_status and test_status:
        print("- Raw text splitting done for training and test data")
