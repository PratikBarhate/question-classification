from qc.ml import train
from qc.ml import test
import sys

if len(sys.argv) != 4:
    print("Error: Expected one argument after module -> root path of the project")
else:
    command = sys.argv[1]
    ml_algo = sys.argv[2]
    root_path = sys.argv[3]
    if command == "train":
        train.execute(root_path, ml_algo)
    elif command == "test":
        test.execute(root_path, ml_algo)
    else:
        print("\n ** Error in initializing function from smv module. Invalid command")
