from qc.dataprep.text_features import text_ft_arr
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
    wflag, w_ft = text_ft_arr(data_type, rp, "word", ml_algo, cat_type)
    pflag, p_ft = text_ft_arr(data_type, rp, "pos", ml_algo, cat_type)
    tflag, t_ft = text_ft_arr(data_type, rp, "tag", ml_algo, cat_type)
    dflag, d_ft = text_ft_arr(data_type, rp, "dep", ml_algo, cat_type)
    sflag, s_ft = text_ft_arr(data_type, rp, "shape", ml_algo, cat_type)
    aflag, a_ft = text_ft_arr(data_type, rp, "alpha", ml_algo, cat_type)
    stflag, st_ft = text_ft_arr(data_type, rp, "stop", ml_algo, cat_type)
    nflag, n_ft = text_ft_arr(data_type, rp, "ner", ml_algo, cat_type)
    x_all_ft = hstack((w_ft, p_ft, t_ft, d_ft, s_ft, a_ft, st_ft, n_ft))
    return x_all_ft
