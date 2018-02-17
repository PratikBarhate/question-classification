from qc.utils.file_ops import read_obj, write_obj
from sklearn.feature_extraction.text import CountVectorizer


def text_ft_vec(data_type: str, rp: str, prop_type: str, ml_algo: str, cat_type: str):
    """
    :argument:
        :param data_type: String either `training` or `test`.
        :param rp: Absolute path of the root directory of the project.
        :param prop_type:  Natural language property either `word` (from spaCy) or `ner` (from StanfordNER).
        :param ml_algo: Machine algorithm for which the dataprep is running.
        :param cat_type: Type of categorical class `coarse` or any of the 6 main classes.
                                        (`abbr` | `desc` | `enty` | `hum` | `loc` | `num`)
    :return:
        boolean_flag: True for successful operation.
        text_ft: feature vectorized to be used in ML algorithms.
    """
    # TODO check proptype should be one among the list which is expected
    # TODO add this check in each init file
    if prop_type == "word":
        flag, doc_obj = read_obj("{1}_{0}_doc".format(data_type, cat_type), rp)
        if flag:
            text_data = []
            for doc in doc_obj:
                text_data.append(doc.text)
            vflag, vectorizer = get_vect(data_type, rp, prop_type, ml_algo, cat_type, text_data)
            if vflag:
                text_ft = vectorizer.transform(text_data)
                return True, text_ft
            else:
                return False
        else:
            return False
    elif prop_type == "ner":
        flag, ner_l = read_obj("{1}_{0}_ner".format(data_type, cat_type), rp)
        if flag:
            vflag, vectorizer = get_vect(data_type, rp, prop_type, ml_algo, cat_type, ner_l)
            if vflag:
                text_ft = vectorizer.transform(ner_l)
                return True, text_ft
            else:
                return False
        else:
            return False
    else:
        print("Error: Invalid `prop_type` to function `text_ft_vec`")
        return False


def get_vect(data_type: str, rp: str, prop_type: str,  ml_algo: str, cat_type: str, text_data):
    """
    :argument:
        :param data_type: String either `training` or `test`.
        :param rp: Absolute path of the root directory of the project.
        :param prop_type:  Natural language property either `word` (from spaCy) or `ner` (from StanfordNER).
        :param ml_algo: Machine algorithm for which the dataprep is running.
        :param cat_type: Type of categorical class `coarse` or any of the 6 main classes.
                                        (`abbr` | `desc` | `enty` | `hum` | `loc` | `num`)
        :param text_data: Data on which CountVectorizer is fitted on while training.
    :return:
        boolean_flag: True for successful operation.
        count_vec: CountVectorizer object.
    """
    if data_type == "training":
        count_vec = CountVectorizer(ngram_range=(3, 5)).fit(text_data)
        wflag = write_obj(count_vec, "{0}_{1}_vec".format(cat_type, prop_type), rp + "/{0}".format(ml_algo))
        return wflag, count_vec
    elif data_type == "test":
        rflag, count_vec = read_obj("{0}_{1}_vec".format(cat_type, prop_type), rp + "/{0}".format(ml_algo))
        return rflag, count_vec
