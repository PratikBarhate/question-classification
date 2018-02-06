from qc import pre_processing
from qc.utils.file_ops import write_obj
from qc.nlp.proc_coarse import com_annotations
from multiprocessing.pool import ThreadPool
import datetime


def coarse_computations(data_type: str, rp: str):
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
        doc_w_flag = write_obj(doc_annot, "coarse_nlp_{0}_doc".format(data), rp)
        if doc_w_flag:
            print("- Computing annotations for {0} data done.".format(data))
            return True
        else:
            print("\n- ERROR: While writing annotations for {0} data.".format(data))
            return False
    else:
        print("\n- ERROR: While computing annotations for {0} data.".format(data))
        return False


def main(project_root_path: str):
    # ensures pre_processing in done
    pre_processing.main(project_root_path)
    # start timer
    start = datetime.datetime.now().timestamp()
    print("\n* NLP for coarse classes")
    # Create two threads for processing training and test raw files
    pool = ThreadPool(processes=2)
    # start the threads and wait for them to finish
    train_result = pool.apply_async(coarse_computations, args=["training", project_root_path])
    test_result = pool.apply_async(coarse_computations, args=["test", project_root_path])
    train_val = train_result.get()
    test_val = test_result.get()
    if not train_val:
        print("- Error: In computing annotations for training data")
    if not test_val:
        print("- Error: In computing annotations for test data")
    if train_val and test_val:
        # timer for end time
        end = datetime.datetime.now().timestamp()
        total = datetime.datetime.utcfromtimestamp(end - start)
        print("- NLP : Done in {0}h {1}m {2}s".format(total.hour, total.minute, total.second))


if __name__ == '__main__':
    # specify the Project root path as per the system used
    main("/Users/tkmahxk/Pratik/Study/Projects/question-classification")
