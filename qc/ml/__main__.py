import sys

from qc.ml import test
from qc.ml import train
from qc.ml import nn
from qc.ml import api

if len(sys.argv) != 4:
    print("Error: Expected one argument after module -> root path of the project")
else:
    command = sys.argv[1]
    ml_algo = sys.argv[2]
    root_path = sys.argv[3]
    if command == "train" and ml_algo == "nn":
        nn.train(root_path)
    elif command == "test" and ml_algo == "nn":
        nn.test(root_path)
    elif command == "train" and ml_algo != "nn":
        train.execute(root_path, ml_algo)
    elif command == "test" and ml_algo != "nn":
        test.execute(root_path, ml_algo)
    elif command == "api":
        if ml_algo == "nn":
            print("\n ** Error in initializing function from ml module. API not supported for nn model.")
        else:
            api.start(root_path, ml_algo)
    else:
        print("\n ** Error in initializing function from ml module. Invalid command")
