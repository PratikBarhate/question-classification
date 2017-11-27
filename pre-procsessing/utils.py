import re


# removing special characters from sentence##
def preprocess(raw_sentence):
    sentence = re.sub(r'[^a-zA-Z0-9\s]', r'', raw_sentence)
    return sentence


def remove_extra_spaces(raw_sentence):
    return re.sub(r'\s\s+', r' ', raw_sentence)
