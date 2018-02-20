from qc.pre_processing.raw_processing import dataset_raw_prep
from multiprocessing.pool import ThreadPool


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
