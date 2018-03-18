from qc.nlp.process_data import execute
import sys

if len(sys.argv) != 2:
    print("Error: Expected one argument after module -> root path of the project")
else:
    execute(sys.argv[1])
