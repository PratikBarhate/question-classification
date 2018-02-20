from qc.nlp.process_data import *
from qc import pre_processing
from multiprocessing.pool import ThreadPool
import datetime
import time


def main(project_root_path: str):
    # ensures pre_processing in done
    pre_processing.main(project_root_path)
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


if __name__ == "__main__":
    # specify the Project root path as per the system used
    main("Absolute Path of the project")
