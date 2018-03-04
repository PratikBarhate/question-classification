from qc.svm import train
from qc.svm import test
import sys


if len(sys.argv) != 3:
    print("Error: Expected one argument after module -> root path of the project")
else:
    command = sys.argv[1]
    root_path = sys.argv[2]
    if command == "train":
        train.execute(root_path)
    elif command == "test":
        # TODO hard coded `svm` remove it
        test.get_predictions(root_path, "svm")
    else:
        print("\n ** Error in initializing function from smv module. Invalid command")
