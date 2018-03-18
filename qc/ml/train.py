from qc.utils.file_ops import read_file, write_obj
from qc.dataprep.feature_stack import get_ft_obj
from qc.pre_processing.raw_processing import remove_endline_char
from multiprocessing.pool import ThreadPool
from sklearn import linear_model
from sklearn import svm
import datetime
import time


def train_one_node(rp: str, cat_type: str, ml_algo: str):
    """
    Gets data in the form of sparse matrix from `qc.dataprep.feature_stack` module
    which is ready for use in a machine learning model. Using the data trains a ml node
    and serialize the trained object to the secondary memory (hard-disk).

    :argument:
        :param rp: Absolute path of the root directory of the project.
        :param cat_type: Type of categorical class `coarse` or any of the 6 main classes.
                                        (`abbr` | `desc` | `enty` | `hum` | `loc` | `num`)
        :param ml_algo: The type of machine learning models to be used. (svm | lr | linear_svm)
    :return:
        boolean_flag: True for successful operation.
        model: trained SVC model
    """
    x_ft = get_ft_obj("training", rp, ml_algo, cat_type)
    rf, labels = read_file("{0}_classes_training".format(cat_type), rp)
    y_lb = [remove_endline_char(c).strip() for c in labels]
    machine = None
    # -----------------------------------Experimental code--------------------------------------------------------------
    # 1. This is the part where you can experiment and play with the parameters.
    # 2. If you want to add more models or combinations, you just need to add a `elif` condition and
    #    provide the condition value in argument from the shell. e.g `train svm`,
    #    here `svm` will be in the variable {ml_algo}.

    if ml_algo == "svm":
        machine = svm.SVC()
    elif ml_algo == "linear_svm":
        machine = svm.LinearSVC()
    elif ml_algo == "lr":
        machine = linear_model.LogisticRegression(solver="newton-cg")
    else:
        print("- Error while training {0} model. {0} is unexpected ML algorithm".format(ml_algo))

    # Parameter tuning ends here.
    # ------------------------------------------------------------------------------------------------------------------
    model = machine.fit(x_ft, y_lb)
    mw_flag = write_obj(model, "{0}_model".format(cat_type), rp + "/{0}".format(ml_algo))
    if mw_flag:
        print("- Training done for {0} model of {1}".format(cat_type, ml_algo))
        return True
    else:
        print("- Error in writing trained {0} model of {1}".format(cat_type, ml_algo))
        return False


def execute(project_root_path: str, ml_algo: str):
    """
    Starts 7 threads to train each of Machine Learning models.
    coarse|abbr|desc|enty|hum|loc|num

    :argument
        :param project_root_path: Absolute Path of the project
        :param ml_algo: The type of machine learning models to be used. (svm | lr | linear_svm)
    :return:
        None
    """
    start_svm = datetime.datetime.now().timestamp()
    print("\n* Training started - {0} model".format(ml_algo))
    pool = ThreadPool(processes=7)
    coarse_result = pool.apply_async(train_one_node, args=[project_root_path, "coarse", ml_algo])
    # add delay to avoid conflicts from multiple threads creating the same directory
    time.sleep(5)
    abbr_result = pool.apply_async(train_one_node, args=[project_root_path, "abbr", ml_algo])
    desc_result = pool.apply_async(train_one_node, args=[project_root_path, "desc", ml_algo])
    enty_result = pool.apply_async(train_one_node, args=[project_root_path, "enty", ml_algo])
    hum_result = pool.apply_async(train_one_node, args=[project_root_path, "hum", ml_algo])
    loc_result = pool.apply_async(train_one_node, args=[project_root_path, "loc", ml_algo])
    num_result = pool.apply_async(train_one_node, args=[project_root_path, "num", ml_algo])
    abbr_val = abbr_result.get()
    desc_val = desc_result.get()
    enty_val = enty_result.get()
    hum_val = hum_result.get()
    loc_val = loc_result.get()
    num_val = num_result.get()
    coarse_val = coarse_result.get()
    if not coarse_val:
        print("- Error while training coarse classifier model")
    if not abbr_val:
        print("- Error while training abbr classifier model")
    if not desc_val:
        print("- Error while training desc classifier model")
    if not enty_val:
        print("- Error while training enty classifier model")
    if not hum_val:
        print("- Error while training hum classifier model")
    if not loc_val:
        print("- Error while training loc classifier model")
    if not num_val:
        print("- Error while training num classifier model")
    if coarse_val and abbr_val and desc_val and enty_val and hum_val and loc_val and num_val:
        end_svm = datetime.datetime.now().timestamp()
        total_svm = datetime.datetime.utcfromtimestamp(end_svm - start_svm)
        print("- Training done : {3} models in {0}h {1}m {2}s"
              .format(total_svm.hour, total_svm.minute, total_svm.second, ml_algo))
