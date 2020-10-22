import numpy
from sklearn.feature_extraction.text import CountVectorizer

from qc.utils.file_ops import read_obj, write_obj


def text_ft_arr(data_type: str, rp: str, prop_type: str, ml_algo: str, cat_type: str, data: list):
    """
    This method reads the list of `doc` objects written to secondary memory by NLP process.
    For the list of objects gets the Word Vectorizer to convert text data to numeric data and transforms
    the list of text data to vectorized features.

    :argument:
        :param data_type: String either `training`, `test` or `api`.
        :param rp: Absolute path of the root directory of the project.
        :param prop_type: Natural language property either `word` | `lemma` | `pos` | `tag` | `dep` |
                          `shape` | `alpha` | `stop` | `ner` | (from spaCy)
        :param ml_algo: Machine algorithm for which the dataprep is running.
        :param cat_type: Type of categorical class `coarse` or any of the 6 main classes.
                         (`abbr` | `desc` | `enty` | `hum` | `loc` | `num`)
        :param data: if data_type='api' then provide the list of Docs here (does not read them from file)
    :return:
        boolean_flag: True for successful operation.
        text_ft: feature vectorized to be used in ML algorithms.
    """
    list_doc_prop = ["word", "lemma", "pos", "tag", "dep", "shape", "alpha", "stop", "ner"]
    if prop_type in list_doc_prop:
        if data_type == "api":
            flag = True
            doc_list_obj = data
        elif data_type == "training":
            flag, doc_list_obj = read_obj("{1}_{0}_doc".format(data_type, cat_type), rp)
        else:
            flag, doc_list_obj = read_obj("coarse_{0}_doc".format(data_type), rp)
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
    else:
        print("- Error: Invalid `prop_type` to function `text_ft_vec`")
        return False


def get_vect(data_type: str, rp: str, prop_type: str, ml_algo: str, cat_type: str, text_data):
    """
    This method takes the list of text data and fits the Word Vectorizer (CountVectorizer) over the list of text data.

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
    # --------------------------------------------Experimental code-----------------------------------------------------
    # Other word embeddings technique can also be tried out - e.g GloVe
    if data_type == "training":
        count_vec = CountVectorizer(ngram_range=(1, 2)).fit(text_data)
        wflag = write_obj(count_vec, "{0}_{1}_vec".format(cat_type, prop_type), rp + "/{0}".format(ml_algo))
        return wflag, count_vec
    elif data_type == "test" or data_type == "api":
        rflag, count_vec = read_obj("{0}_{1}_vec".format(cat_type, prop_type), rp + "/{0}".format(ml_algo))
        return rflag, count_vec
    else:
        print("Error: Wrong `data_type` param to function `dataprep.text.get_vect`")
        return False

    # Word vectorization ends here.
    # ------------------------------------------------------------------------------------------------------------------


def get_info_doc(prop: str, doc_obj):
    """
    'Doc' data type of spaCy lib keeps the computed information in the form of token for each of the word in a sentence.
    But we need the information as a single string analogous to the original sentence. Hence, this method gets the
    tokens and joins the information of each word into one string.

    :argument
        :param prop: Natural language property either `word` | `lemma` | `pos` | `tag` | `dep` |
                     `shape` | `alpha` | `stop` | `ner` | (from spaCy lib)
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
    elif prop == "ner":
        for doc in doc_obj:
            ner_list = []
            for t in doc:
                if t.ent_type_ == "":
                    ner_list.append("0")
                else:
                    ner_list.append(t.ent_type_)
            data.append(" ".join(ner_list))
    return data
