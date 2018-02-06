from qc import nlp
import sys


if len(sys.argv) != 2:
    print("Error: Expected one argument after module -> root path of the project")
else:
    nlp.main(sys.argv[1])
