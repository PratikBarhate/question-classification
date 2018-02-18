from qc.utils.file_ops import read_obj


def ann_hashed_ft(data_type: str, rp: str, cat_type: str):
    """
    :argument
        :param data_type: String either `training` or `test`.
        :param rp: Absolute path of the root directory of the project.
        :param cat_type: Type of categorical class `coarse` or any of the 6 main classes.
                                        (`abbr` | `desc` | `enty` | `hum` | `loc` | `num`)
    :return:
        boolean_flag: True for successful operation.
        lemma_ft: stem form of the word as hashed feature.
        pos_ft: Past of Speech tags as hashed feature.
        tag_ft: In-dept tags of word as hashed feature.
        dep_ft: Dependency tree as hashed feature.
        shape_ft: Shape of the word as hashed feature.
        alpha_ft: Is the word an alphabet or numeric string as boolean data.
        stop_ft: Is the word stop word or not as boolean data.
    """
    flag, doc_list_obj = read_obj("{1}_{0}_doc".format(data_type, cat_type), rp)
    if flag:
        lemma_ft = []
        pos_ft = []
        tag_ft = []
        dep_ft = []
        shape_ft = []
        alpha_ft = []
        stop_ft = []
        for doc in doc_list_obj:
            lemma_ft.append(doc.lemma)
            pos_ft.append(doc.pos)
            tag_ft.append(doc.tag)
            dep_ft.append(doc.dep)
            shape_ft.append(doc.shape)
            alpha_ft.append(doc.is_alpha)
            stop_ft.append(doc.is_stop)
        return True, lemma_ft, pos_ft, tag_ft, dep_ft, shape_ft, alpha_ft, stop_ft
    else:
        return False
