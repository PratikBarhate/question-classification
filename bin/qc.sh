#!/bin/bash

#Find the script file parent directory (project home)
pushd . > /dev/null
SCRIPT_DIRECTORY="${BASH_SOURCE[0]}"
while([ -h "${SCRIPT_DIRECTORY}" ])
do
  cd "`dirname "${SCRIPT_DIRECTORY}"`"
  SCRIPT_DIRECTORY="$(readlink "`basename "${SCRIPT_DIRECTORY}"`")"
done
cd "`dirname "${SCRIPT_DIRECTORY}"`" > /dev/null
SCRIPT_DIRECTORY="`pwd`"
popd  > /dev/null
APP_HOME="`dirname "${SCRIPT_DIRECTORY}"`"


num_of_arg=$#


if [ ${num_of_arg} -eq 0 ]
then
  echo "At least one argument required. Following are the expected arguments:-"
  echo "1. pre-process (to only split the raw data and their classes in different text file)"
  echo "2. nlp (to complete all the NLP operations)"
else
  export PYTHONPATH="${APP_HOME}"
  if [ ${1} == "pre-process" ]
  then
    python -m qc.pre_processing
  elif [ ${1} == "nlp" ]
  then
    python -m qc.nlp.__init__.py
  else
    echo "Invalid first argument. ${1} as first argument is unexpected."
  fi
fi
