from qc import pre_processing
import sys


if len(sys.argv) != 2:
    print("Error: Expected one argument after module -> root path of the project")
else:
    pre_processing.main(sys.argv[1])
