from qc import pre_processing
from qc.utils.file_ops import write_obj
from qc.nlp.proc_coarse import com_annotations, com_ner
from multiprocessing.pool import ThreadPool
import datetime
import time


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
        ner_w_flag = write_obj(ner_tags, "coarse_nlp_{0}_ner".format(data), rp)
        if ner_w_flag:
            print("- Computing NER tags for {0} data done.".format(data))
            return True
        else:
            print("\n- ERROR: While writing NER tags for {0} data.".format(data))
            return False
    else:
        print("\n- ERROR: While computing NER tags for {0} data.".format(data))
        return False


def main(project_root_path: str):
    # ensures pre_processing in done
    pre_processing.main(project_root_path)
    # start timer
    start = datetime.datetime.now().timestamp()
    print("\n* NLP")
    # Create 4 threads for processing training and test raw files
    # 2 Threads to compute annotations from spaCy lib
    # 2 Threads to compute NER tags from StanfordNER Client
    pool = ThreadPool(processes=4)
    # start the threads and wait for them to finish
    train_ann_result = pool.apply_async(coarse_ann_computations, args=["training", project_root_path])
    # wait for 1 second to avoid creation of same directory by different threads
    time.sleep(1)
    train_ner_result = pool.apply_async(coarse_ner_computations, args=["training", project_root_path])
    test_ann_result = pool.apply_async(coarse_ann_computations, args=["test", project_root_path])
    # wait for 1 second to avoid creation of same directory by different threads
    time.sleep(1)
    test_ner_result = pool.apply_async(coarse_ner_computations, args=["test", project_root_path])
    test_ann_val = test_ann_result.get()
    test_ner_val = test_ner_result.get()
    train_ann_val = train_ann_result.get()
    train_ner_val = train_ner_result.get()
    if not train_ann_val:
        print("- Error: In computing annotations for training data")
    if not test_ann_val:
        print("- Error: In computing annotations for test data")
    if not train_ner_val:
        print("- Error: In computing NER tags for training data")
    if not test_ner_val:
        print("- Error: In computing NER tags for test data")
    if train_ann_val and test_ann_val and train_ner_val and test_ner_val:
        # timer for end time
        end = datetime.datetime.now().timestamp()
        total = datetime.datetime.utcfromtimestamp(end - start)
        print("- NLP : Done in {0}h {1}m {2}s".format(total.hour, total.minute, total.second))


if __name__ == "__main__":
    # specify the Project root path as per the system used
    main("Absolute Path of the project")
