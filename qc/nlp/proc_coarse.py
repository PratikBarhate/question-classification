import spacy
from qc.utils.file_ops import read_file


def com_annotations(data_type: str):
    """
    Function gets all the annotations, Lemma, POS, IS_STOPWORD etc., in the form of spaCy doc container for all the
    rows (lines) in the text data and saves the object of list of the doc (container) for each line.
    # doc = spaCy container - [https://spacy.io/api/doc]

    :argument:
        :param data_type: String either `training` or `test`
    :return:
        boolean_flag: True for successful operation.
        all_annotations: List of doc (spaCy containers) of all the lines in the data.
    """
    nlp = spacy.load("en_core_web_lg")
    all_annotations = []
    read_flag, file = read_file("raw_sentence_{0}".format(data_type))
    if read_flag:
        for line in file:
            doc = nlp(line)
            all_annotations.append(doc)
        file.close()
        return True, all_annotations
    else:
        return False
