from qc.pre_processing.raw_processing import read_raw_data, clean_sentences
from qc.utils.file_ops import write_str_file
from multiprocessing.pool import ThreadPool


def dataset_raw_prep(data_type, rp: str):
    """
    :argument:
        :param data_type: String either `training` or `test`
        :param rp: Absolute path of the root directory of the project
    :Execution:
        Calls various raw_processing functions on the given raw text data
    :return:
        boolean flag: True for successful operation
    """
    data = "training" if data_type == "training" else "test"
    flag, coarse_class, fine_class, questions = read_raw_data("{0}_data".format(data), rp)
    if flag:
        q_clean = clean_sentences(questions)
        c = write_str_file(coarse_class, "coarse_classes_{0}".format(data), rp)
        f = write_str_file(fine_class, "fine_classes_{0}".format(data), rp)
        q = write_str_file(q_clean, "raw_sentence_{0}".format(data), rp)
        if not q:
            print("- Error while writing questions file for " + data)
            return False
        if not c:
            print("- Error while writing coarse class file for " + data)
            return False
        if not f:
            print("- Error while writing fine class file for " + data)
            return False
        return True
    else:
        print("- Error is reading and splitting " + data + " data")
        return False


def main(project_root_path: str):
    print("\n* Raw Data Processing")
    # Create two threads for processing training and test raw files
    pool = ThreadPool(processes=2)
    # start the threads and wait for them to finish
    train_result = pool.apply_async(dataset_raw_prep, args=["training", project_root_path])
    test_result = pool.apply_async(dataset_raw_prep, args=["test", project_root_path])
    train_val = train_result.get()
    test_val = test_result.get()
    if not train_val:
        print("- Error: In text splitting for training data")
    if not test_val:
        print("- Error: In text splitting for test data")
    if train_val and test_val:
        print("- Raw text splitting done for training and test data")


if __name__ == "__main__":
    # specify the Project root path as per the system used
    main("Absolute Path of the project")
