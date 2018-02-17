from qc import dataprep
import sys


if len(sys.argv) != 2:
    print("Error: Expected one argument after module -> root path of the project")
else:
    dataprep.main(sys.argv[1])
