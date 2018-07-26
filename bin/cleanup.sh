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
if [ ! ${num_of_arg} -eq 1 ]
then
  echo "One argument expected. Given ${num_of_arg}"
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
elif [ ${1} == "models" ]
then
  rm -rf "${APP_HOME}/svm"
  rm -rf "${APP_HOME}/lr"
  rm -rf "${APP_HOME}/linear_svm"
elif [ ${1} == "nlp" ]
then
  rm -rf "${APP_HOME}/common_data"
elif [ ${1} == "nn" ]
then
  rm -rf "${APP_HOME}/nn"
elif [ ${1} == "svm" ]
then
  rm -rf "${APP_HOME}/svm"
elif [ ${1} == "lr" ]
then
  rm -rf "${APP_HOME}/lr"
elif [ ${1} == "linear_svm" ]
then
  rm -rf "${APP_HOME}/linear_svm"
else
  echo "Unexpected argument: ${1}"
fi

