#!/bin/bash

#Find the script file parent directory (project home)
pushd . > /dev/null
SCRIPT_DIRECTORY="${BASH_SOURCE[0]}"
while([ -h "${SCRIPT_DIRECTORY}" ])
do
  cd "$(dirname "${SCRIPT_DIRECTORY}")"
  SCRIPT_DIRECTORY="$(readlink "$(basename "${SCRIPT_DIRECTORY}")")"
done
cd "$(dirname "${SCRIPT_DIRECTORY}")" > /dev/null
SCRIPT_DIRECTORY="$(pwd)"
popd  > /dev/null
APP_HOME="$(dirname "${SCRIPT_DIRECTORY}")"


# check for the number arguments provided are as expected or not
num_of_arg=$#
if [ ${num_of_arg} -eq 0 ]
then
  echo "At least one argument expected. Given ${num_of_arg}"
  exit 1
fi


# Clean the directories as per the argument
if [ ${1} == "all" ]
then
  rm -rf "${APP_HOME}/common_data"
  rm -rf "${APP_HOME}/svm"
  rm -rf "${APP_HOME}/lr"
  rm -rf "${APP_HOME}/linear_svm"
  rm -rf "${APP_HOME}/nn"
elif [ ${1} == "all_models" ]
then
  rm -rf "${APP_HOME}/svm"
  rm -rf "${APP_HOME}/lr"
  rm -rf "${APP_HOME}/linear_svm"
elif [ ${1} == "nlp" ]
then
  rm -rf "${APP_HOME}/common_data"
elif [ ${1} == "model" ]
then
  if [ ! ${num_of_arg} -eq 2 ]
  then
    echo "One more argument expected with 'model', 'ml_algo_model' - model name."
    exit 1
  fi
  rm -rf "${APP_HOME}/${2}"
else
  echo "Unexpected argument: ${1}"
fi

