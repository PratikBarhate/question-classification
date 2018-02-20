from qc.utils.file_ops import write_obj
from qc.nlp.proc_coarse import com_annotations, com_ner
from qc.nlp.proc_fine import sep_lang_prop


def coarse_ann_computations(data_type: str, rp: str):
    """
    :argument:
        :param data_type: String either `training` or `test`
        :param rp: Absolute path of the root directory of the project
    :return:
        boolean_flag: True for successful operation.
    """
    data = "training" if data_type == "training" else "test"
    doc_flag, doc_annot = com_annotations(data, rp)
    if doc_flag:
        doc_w_flag = write_obj(doc_annot, "coarse_{0}_doc".format(data), rp)
        if doc_w_flag:
            print("- Computing annotations for {0} data done.".format(data))
            return True
        else:
            print("\n- ERROR: While writing annotations for {0} data.".format(data))
            return False
    else:
        print("\n- ERROR: While computing annotations for {0} data.".format(data))
        return False


def coarse_ner_computations(data_type: str, rp: str):
    """
    :argument:
        :param data_type: String either `training` or `test`
        :param rp: Absolute path of the root directory of the project
    :return:
        boolean_flag: True for successful operation.
    """
    data = "training" if data_type == "training" else "test"
    ner_flag, ner_tags = com_ner(data, rp)
    if ner_flag:
        ner_w_flag = write_obj(ner_tags, "coarse_{0}_ner".format(data), rp)
        if ner_w_flag:
            print("- Computing NER tags for {0} data done.".format(data))
            return True
        else:
            print("\n- ERROR: While writing NER tags for {0} data.".format(data))
            return False
    else:
        print("\n- ERROR: While computing NER tags for {0} data.".format(data))
        return False


def fine_prop_separation(data_type: str, rp: str, prop_type: str):
    """
    :argument:
        :param data_type: String either `training` or `test`
        :param rp: Absolute path of the root directory of the project
        :param prop_type: Natural language property either `doc` (from spaCy) or `ner` (from StanfordNER)
    :return:
        boolean_flag: True for successful operation.
    """
    data = "training" if data_type == "training" else "test"
    prop_flag, abbr_prop, desc_prop, enty_prop, hum_prop, loc_prop, num_prop = sep_lang_prop(data, rp, prop_type)
    if prop_flag:
        wf_1 = write_obj(abbr_prop, "abbr_{0}_{1}".format(data, prop_type), rp)
        wf_2 = write_obj(desc_prop, "desc_{0}_{1}".format(data, prop_type), rp)
        wf_3 = write_obj(enty_prop, "enty_{0}_{1}".format(data, prop_type), rp)
        wf_4 = write_obj(hum_prop, "hum_{0}_{1}".format(data, prop_type), rp)
        wf_5 = write_obj(loc_prop, "loc_{0}_{1}".format(data, prop_type), rp)
        wf_6 = write_obj(num_prop, "num_{0}_{1}".format(data, prop_type), rp)
        if wf_1 and wf_2 and wf_3 and wf_4 and wf_5 and wf_6:
            print("- Separating {1} tags for {0} data done.".format(data, prop_type))
            return True
        else:
            print("\n- ERROR: While writing {1} tags for {0} data.".format(data, prop_type))
            return False
    else:
        print("\n- ERROR: While computing {1} tags for {0} data.".format(data, prop_type))
        return False
