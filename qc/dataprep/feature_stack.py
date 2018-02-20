from qc.dataprep.numeric import ann_hashed_ft
from qc.dataprep.text import text_ft_vec
from numpy import hstack


def get_ft_obj(data_type: str, rp: str, ml_algo: str, cat_type: str):
    """
    :argument:
        :param data_type: String either `training` or `test`.
        :param rp: Absolute path of the root directory of the project.
        :param ml_algo: Machine algorithm for which the dataprep is running.
        :param cat_type: Type of categorical class `coarse` or any of the 6 main classes.
                                        (`abbr` | `desc` | `enty` | `hum` | `loc` | `num`)
    :return:
        boolean_flag: True for successful operation.
        x_all_ft: numpy array of features to be feed to a Machine Learning algorithm.
    """
    all_num_ft = ann_hashed_ft(data_type, rp, cat_type)
    x_num_ft = hstack(tuple(all_num_ft))
    # TODO check results after standardizing num ft once a ml algorithm is ready to test
    w_ft = text_ft_vec(data_type, rp, "word", ml_algo, cat_type)
    n_ft = text_ft_vec(data_type, rp, "ner", ml_algo, cat_type)
    x_text_ft = hstack((w_ft, n_ft))
    x_all_ft = hstack((x_num_ft, x_text_ft))
    return x_all_ft
