import spacy

from qc.pre_processing.raw_processing import remove_endline_char, remove_space_before_apost, remove_extra_spaces, \
    pre_process
from qc.utils.file_ops import read_file

nlp = None


def load_spacy():
    global nlp
    if nlp is None:
        nlp = spacy.load("en_core_web_lg")


def com_annotations(data_type: str, rp: str):
    """
    This method computes all the annotations, Lemma, POS, IS_STOPWORD etc., in the form of spaCy doc container
    for all the rows (lines) in the text data and saves the object of list of the doc (container) for each line.
    # doc = spaCy container - [https://spacy.io/api/doc]

    :argument:
        :param data_type: String either `training` or `test`
        :param rp: Absolute path of the root directory of the project
    :return:
        boolean_flag: True for successful operation.
        all_annotations: List of doc (spaCy containers) of all the lines in the data.
    """
    load_spacy()

    all_annotations = []
    read_flag, file = read_file("raw_sentence_{0}".format(data_type), rp)
    if read_flag:
        for line in file:
            doc = nlp(remove_endline_char(line))
            all_annotations.append(doc)
        file.close()
        return True, all_annotations
    else:
        return False


def com_annotations_param(text: str):
    """
    Same as com_annotations() above but the input is not a file but the function parameter `text`.

    :argument:
        :param text Input text to be annotated.
    :return:
        boolean_flag: True for successful operation.
        annotation: doc (spaCy container)
    """
    load_spacy()

    sanitized_text = remove_endline_char(
        remove_space_before_apost(
            remove_extra_spaces(
                pre_process(
                    text
                )
            )
        )
    )

    doc = nlp(text)

    return True, doc
