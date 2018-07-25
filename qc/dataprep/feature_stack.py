from scipy.sparse import hstack

from qc.dataprep.text_features import text_ft_arr


def get_ft_obj(data_type: str, rp: str, ml_algo: str, cat_type: str):
    """
    This method gets the vectorized features and stacks them horizontally.

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
    # NOTE:
    # 1. Part of Speech (pos) is same as `tag` hence no needed, and reduces accuracy.
    #    More information can be found here - [https://spacy.io/api/token]
    # 2. Direct list words, is not a good feature,
    #    instead lemma (root form) of the word is more useful as a feature.
    # 3. Is alphabet or not feature is reducing the accuracy by a bit,
    #    hence not used for now.

    # -------------------------------------------Experimental code------------------------------------------------------
    # Here you can select and tune feature stack.

    # p_ft = text_ft_arr(data_type, rp, "pos", ml_algo, cat_type)[1]
    # w_ft = text_ft_arr(data_type, rp, "word", ml_algo, cat_type)[1]
    # a_ft = text_ft_arr(data_type, rp, "alpha", ml_algo, cat_type)[1]
    l_ft = text_ft_arr(data_type, rp, "lemma", ml_algo, cat_type)[1]
    t_ft = text_ft_arr(data_type, rp, "tag", ml_algo, cat_type)[1]
    d_ft = text_ft_arr(data_type, rp, "dep", ml_algo, cat_type)[1]
    s_ft = text_ft_arr(data_type, rp, "shape", ml_algo, cat_type)[1]
    st_ft = text_ft_arr(data_type, rp, "stop", ml_algo, cat_type)[1]
    n_ft = text_ft_arr(data_type, rp, "ner", ml_algo, cat_type)[1]
    x_all_ft = hstack([t_ft, d_ft, n_ft, st_ft, l_ft, s_ft])

    # Feature stack  ends here.
    # ------------------------------------------------------------------------------------------------------------------
    return x_all_ft
