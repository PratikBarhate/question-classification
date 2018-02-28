from qc.utils.file_ops import write_obj
from qc.nlp.proc_coarse import com_annotations, com_ner
from qc.nlp.proc_fine import sep_lang_prop
from qc.pre_processing import raw_processing
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


def execute(project_root_path: str):
    """
    :argument
        :param project_root_path: Absolute Path of the project
    :return:
        None
    """
    # ensures pre_processing in done
    raw_processing.execute(project_root_path)
    # start timer
    start_nlp = datetime.datetime.now().timestamp()
    print("\n* NLP")
    # Create 4 threads for processing training and test raw files
    # 2 Threads to compute annotations from spaCy lib
    # 2 Threads to compute NER tags from StanfordNER Client
    pool = ThreadPool(processes=4)
    # start the threads and wait for them to finish
    train_ann = pool.apply_async(coarse_ann_computations, args=["training", project_root_path])
    # wait for 1 second to avoid creation of same directory by different threads
    time.sleep(1)
    train_ner = pool.apply_async(coarse_ner_computations, args=["training", project_root_path])
    test_ann = pool.apply_async(coarse_ann_computations, args=["test", project_root_path])
    # wait for 1 second to avoid creation of same directory by different threads
    time.sleep(1)
    test_ner = pool.apply_async(coarse_ner_computations, args=["test", project_root_path])
    test_ann_val = test_ann.get()
    test_ner_val = test_ner.get()
    train_ann_val = train_ann.get()
    train_ner_val = train_ner.get()
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
        end_nlp = datetime.datetime.now().timestamp()
        total_nlp = datetime.datetime.utcfromtimestamp(end_nlp - start_nlp)
        print("- NLP : Done in {0}h {1}m {2}s".format(total_nlp.hour, total_nlp.minute, total_nlp.second))
    # separate the computed Natural language properties for each coarse class (main categories)
    print("\n* Separating NLP properties for each of the Coarse classes (categories)")
    # use the same thread created before
    start_sep = datetime.datetime.now().timestamp()
    train_sep_ann = pool.apply_async(fine_prop_separation, args=["training", project_root_path, "doc"])
    # wait for 1 second to avoid creation of same directory by different threads
    time.sleep(1)
    train_sep_ner = pool.apply_async(fine_prop_separation, args=["training", project_root_path, "ner"])
    test_sep_ann = pool.apply_async(fine_prop_separation, args=["test", project_root_path, "doc"])
    # wait for 1 second to avoid creation of same directory by different threads
    time.sleep(1)
    test_sep_ner = pool.apply_async(fine_prop_separation, args=["test", project_root_path, "ner"])
    train_sep_ann_val = train_sep_ann.get()
    train_sep_ner_val = train_sep_ner.get()
    test_sep_ann_val = test_sep_ann.get()
    test_sep_ner_val = test_sep_ner.get()
    if not train_sep_ann_val:
        print("- Error: In separating annotations for training data")
    if not test_sep_ann_val:
        print("- Error: In separating annotations for test data")
    if not train_sep_ner_val:
        print("- Error: In separating NER tags for training data")
    if not test_sep_ner_val:
        print("- Error: In separating NER tags for test data")
    if train_sep_ann_val and test_sep_ann_val and train_sep_ner_val and test_sep_ner_val:
        # timer for end time
        end_sep = datetime.datetime.now().timestamp()
        total_sep = datetime.datetime.utcfromtimestamp(end_sep - start_sep)
        print("- NLP properties separation : Done in {0}h {1}m {2}s"
              .format(total_sep.hour, total_sep.minute, total_sep.second))
