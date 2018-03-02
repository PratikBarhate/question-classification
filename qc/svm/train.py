from qc.utils.file_ops import read_file, write_obj
from qc.dataprep.feature_stack import get_ft_obj
from qc.pre_processing.raw_processing import remove_endline_char
from multiprocessing.pool import ThreadPool
from sklearn import svm
import datetime
import time


def execute(project_root_path: str):
    """
    Starts 7 threads to train each of SVM models.

    :argument
        :param project_root_path: Absolute Path of the project
    :return:
        None
    """
    start_svm = datetime.datetime.now().timestamp()
    print("\n* Support Vector Machine (SVM) - Training started")
    pool = ThreadPool(processes=7)
    coarse_result = pool.apply_async(train_one_node, args=[project_root_path, "coarse"])
    # add delay to avoid conflicts from multiple threads creating the same directory
    time.sleep(5)
    abbr_result = pool.apply_async(train_one_node, args=[project_root_path, "abbr"])
    desc_result = pool.apply_async(train_one_node, args=[project_root_path, "desc"])
    enty_result = pool.apply_async(train_one_node, args=[project_root_path, "enty"])
    hum_result = pool.apply_async(train_one_node, args=[project_root_path, "hum"])
    loc_result = pool.apply_async(train_one_node, args=[project_root_path, "loc"])
    num_result = pool.apply_async(train_one_node, args=[project_root_path, "num"])
    coarse_val = coarse_result.get()
    abbr_val = abbr_result.get()
    desc_val = desc_result.get()
    enty_val = enty_result.get()
    hum_val = hum_result.get()
    loc_val = loc_result.get()
    num_val = num_result.get()
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
        print("- SVM models : Training done in {0}h {1}m {2}s"
              .format(total_svm.hour, total_svm.minute, total_svm.second))


def train_one_node(rp: str, cat_type: str):
    """
    Gets data in the form of sparse matrix from `qc.dataprep.feature_stack` module
    which is ready for use in a machine learning model. Using the data trains a svm node
    and serialize the trained object to the secondary memory.

    :argument:
        :param rp: Absolute path of the root directory of the project.
        :param cat_type: Type of categorical class `coarse` or any of the 6 main classes.
                                        (`abbr` | `desc` | `enty` | `hum` | `loc` | `num`)
    :return:
        boolean_flag: True for successful operation.
        model: trained SVC model
    """
    x_ft = get_ft_obj("training", rp, "svm", cat_type)
    rf, labels = read_file("{0}_classes_training".format(cat_type), rp)
    y_lb = [remove_endline_char(c).strip() for c in labels]
    machine = svm.SVC(kernel="rbf", C=1.0, gamma=0.0005)
    model = machine.fit(x_ft, y_lb)
    mw_flag = write_obj(model, "{0}_model".format(cat_type), rp + "/svm")
    if mw_flag:
        print("- Training done for {0} model of svm".format(cat_type))
        return True
    else:
        print("- Error in writing trained {0} model of svm".format(cat_type))
        return False
