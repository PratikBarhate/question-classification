import re
from utils import read_operations


# removing special characters from sentence##
def preprocess(raw_sentence):
    sentence = re.sub(r'[^a-zA-Z0-9\s]', r'', raw_sentence)
    return sentence

# remove
def remove_extra_spaces(raw_sentence):
    return re.sub(r'\s\s+', r' ', raw_sentence)


def raw_file_processing():
    file = open(read_operations.read_file(("training_data")), "r")
