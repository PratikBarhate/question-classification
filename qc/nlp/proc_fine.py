from qc.utils.file_ops import read_obj, read_file
from qc.pre_processing.raw_processing import remove_endline_char


def sep_lang_prop(data_type: str, rp: str, prop_type: str):
    """
    Function gets all the Natural language properties, which are pre-computed and separate outs for annotations for
    each coarse classes which will make things easier to train our sub-models
    (model for fine classes given the particular coarse class).

    :argument:
        :param data_type: String either `training` or `test`
        :param rp: Absolute path of the root directory of the project
        :param prop_type: Natural language property either `doc` (from spaCy) or `ner` (from StanfordNER)
    :return:
        boolean_flag: True for successful operation.
        abbr_prop: List of prop (spaCy containers or NER tags) of questions belonging to ABBR coarse class.
        desc_prop: List of prop (spaCy containers or NER tags) of questions belonging to DESC coarse class.
        enty_prop: List of prop (spaCy containers or NER tags) of questions belonging to ENTY coarse class.
        hum_prop: List of prop (spaCy containers or NER tags) of questions belonging to HUM coarse class.
        loc_prop: List of prop (spaCy containers or NER tags) of questions belonging to LOC coarse class.
        num_prop: List of prop (spaCy containers or NER tags) of questions belonging to NUM coarse class.
    """
    abbr_prop = []
    desc_prop = []
    enty_prop = []
    hum_prop = []
    loc_prop = []
    num_prop = []
    read_obj_flag, all_prop_obj = read_obj("coarse_{0}_{1}".format(data_type, prop_type), rp)
    read_file_flag, coarse_classes_file = read_file("coarse_classes_{0}".format(data_type), rp)
    if read_obj_flag and read_file_flag:
        i = 0
        for line in coarse_classes_file:
            coarse_c = remove_endline_char(line).strip()
            if coarse_c == "ABBR":
                abbr_prop.append(all_prop_obj[i])
            elif coarse_c == "DESC":
                desc_prop.append(all_prop_obj[i])
            elif coarse_c == "ENTY":
                enty_prop.append(all_prop_obj[i])
            elif coarse_c == "HUM":
                hum_prop.append(all_prop_obj[i])
            elif coarse_c == "LOC":
                loc_prop.append(all_prop_obj[i])
            elif coarse_c == "NUM":
                num_prop.append(all_prop_obj[i])
            else:
                print("{0} is an unexpected coarse class".format(coarse_c))
            # increment i by one, so that the proper lines are match
            i = i + 1
        # for validation
        if i != len(all_prop_obj):
            print("Something went wrong in mapping the processed annotations.")
            return False
        return True, abbr_prop, desc_prop, enty_prop, hum_prop, loc_prop, num_prop
    else:
        return False
