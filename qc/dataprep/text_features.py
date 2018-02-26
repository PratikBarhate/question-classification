from qc.utils.file_ops import read_obj, write_obj
from sklearn.feature_extraction.text import CountVectorizer
import numpy


def text_ft_arr(data_type: str, rp: str, prop_type: str, ml_algo: str, cat_type: str):
    """
    :argument:
        :param data_type: String either `training` or `test`.
        :param rp: Absolute path of the root directory of the project.
        :param prop_type: Natural language property either `word` | `lemma` | `pos` | `tag` | `dep` |
                          `shape` | `alpha` | `stop` (from spaCy) or `ner` (from StanfordNER).
        :param ml_algo: Machine algorithm for which the dataprep is running.
        :param cat_type: Type of categorical class `coarse` or any of the 6 main classes.
                         (`abbr` | `desc` | `enty` | `hum` | `loc` | `num`)
    :return:
        boolean_flag: True for successful operation.
        text_ft: feature vectorized to be used in ML algorithms.
    """
    list_doc_prop = ["word", "lemma", "pos", "tag", "dep", "shape", "alpha", "stop"]
    if prop_type in list_doc_prop:
        flag, doc_list_obj = read_obj("{1}_{0}_doc".format(data_type, cat_type), rp)
        if flag:
            text_data = get_info_doc(prop_type, doc_list_obj)
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
        print("- Error: Invalid `prop_type` to function `text_ft_vec`")
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
    else:
        print("Error: Wrong `data_type` param to function `dataprep.text.get_vect`")
        return False


def get_info_doc(prop: str, doc_obj):
    """
    :argument
        :param prop: Natural language property either `word` | `lemma` | `pos` | `tag` | `dep` |
                     `shape` | `alpha` | `stop` (from spaCy)
        :param doc_obj: `Doc` container of spaCy library.
    :return:
        data: Natural language annotations/properties as computed by the library as space concatenated string.
    """
    data = []
    if prop == "word":
        for doc in doc_obj:
            data.append(doc.text)
    elif prop == "lemma":
        for doc in doc_obj:
            data.append(" ".join([t.lemma_ for t in doc]))
    elif prop == "pos":
        for doc in doc_obj:
            data.append(" ".join([t.pos_ for t in doc]))
    elif prop == "tag":
        for doc in doc_obj:
            data.append(" ".join([t.tag_ for t in doc]))
    elif prop == "dep":
        for doc in doc_obj:
            data.append(" ".join([t.dep_ for t in doc]))
    elif prop == "shape":
        for doc in doc_obj:
            data.append(" ".join([t.shape_ for t in doc]))
    elif prop == "alpha":
        for doc in doc_obj:
            data.append(" ".join(numpy.array([t.is_alpha for t in doc], dtype=str).tolist()))
    elif prop == "stop":
        for doc in doc_obj:
            data.append(" ".join(numpy.array([t.is_stop for t in doc], dtype=str).tolist()))
    return data
