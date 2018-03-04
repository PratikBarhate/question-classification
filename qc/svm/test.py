from qc.utils.file_ops import read_obj
from qc.dataprep.feature_stack import get_ft_obj
import datetime


def get_predictions(rp: str, ml_algo: str):
    """
    Gets data in the form of sparse matrix from `qc.dataprep.feature_stack` module
    which is ready for use in a machine learning model. Using the test data in `question-classification/dataset`
    tests [[ml_algo]] model which is pre-trained and saved as serialized pickle files.

    :argument
        :param rp: Absolute path of the root directory of the project.
        :param ml_algo: The type of machine learning models to be used. (svm | lr)
    :return:
        pred: List of prediction for each of the test questions (test data).
    """
    start_test = datetime.datetime.now().timestamp()
    print("\n* Testing started - {0} model".format(ml_algo))
    pred = []
    # load all the trained models
    crf, coarse_model = read_obj("coarse_model", rp + "/{0}".format(ml_algo).format(ml_algo))
    arf, abbr_model = read_obj("abbr_model", rp + "/{0}".format(ml_algo))
    drf, desc_model = read_obj("desc_model", rp + "/{0}".format(ml_algo))
    erf, enty_model = read_obj("enty_model", rp + "/{0}".format(ml_algo))
    hrf, hum_model = read_obj("hum_model", rp + "/{0}".format(ml_algo))
    lrf, loc_model = read_obj("loc_model", rp + "/{0}".format(ml_algo))
    nrf, num_model = read_obj("num_model", rp + "/{0}".format(ml_algo))
    if not crf:
        print("- Error in reading coarse {0} model".format(ml_algo))
    if not arf:
        print("- Error in reading abbr {0} model".format(ml_algo))
    if not drf:
        print("- Error in reading desc {0} model".format(ml_algo))
    if not erf:
        print("- Error in reading enty {0} model".format(ml_algo))
    if not hrf:
        print("- Error in reading hum {0} model".format(ml_algo))
    if not lrf:
        print("- Error in reading loc {0} model".format(ml_algo))
    if not nrf:
        print("- Error in reading num {0} model".format(ml_algo))
    if crf and arf and drf and erf and hrf and lrf and nrf:
        print("- Loading the {0} models complete".format(ml_algo))
    else:
        print("- Error in loading the pre-trained model")
        exit(-10)
    c_ft = get_ft_obj("test", rp, ml_algo, "coarse").tocsr()
    a_ft = get_ft_obj("test", rp, ml_algo, "abbr").tocsr()
    d_ft = get_ft_obj("test", rp, ml_algo, "desc").tocsr()
    e_ft = get_ft_obj("test", rp, ml_algo, "enty").tocsr()
    h_ft = get_ft_obj("test", rp, ml_algo, "hum").tocsr()
    l_ft = get_ft_obj("test", rp, ml_algo, "loc").tocsr()
    n_ft = get_ft_obj("test", rp, ml_algo, "num").tocsr()
    print("- DataPrep for test data done.")
    for i in range(0, c_ft.shape[0]):
        c = coarse_model.predict(c_ft[i])[0]
        if c == "ABBR":
            f = abbr_model.predict(a_ft[i])[0]
        elif c == "DESC":
            f = desc_model.predict(d_ft[i])[0]
        elif c == "ENTY":
            f = enty_model.predict(e_ft[i])[0]
        elif c == "HUM":
            f = hum_model.predict(h_ft[i])[0]
        elif c == "LOC":
            f = loc_model.predict(l_ft[i])[0]
        else:
            f = num_model.predict(n_ft[i])[0]
        row_pred = [c, f]
        pred.append(row_pred)
    end_test = datetime.datetime.now().timestamp()
    total_test = datetime.datetime.utcfromtimestamp(end_test - start_test)
    print("- Predicting done : {3} models in {0}h {1}m {2}s"
          .format(total_test.hour, total_test.minute, total_test.second, ml_algo))
    return pred
