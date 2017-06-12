#!/bin/bash
set -ex
export EXP_CONFIG='test'
NOTEBOOK=model_gym.ipynb
if [ -n "$1" ] ; then 
  NOTEBOOK=$1
  shift
fi
if [ -n "$1" ] ; then 
  export EXP_CONFIG=$1
  shift
fi

mkdir -p results
NOTEBOOK_NAME=${NOTEBOOK%.*}
NOTEBOOK_OUT=results/${NOTEBOOK_NAME}_${EXP_CONFIG}.ipynb
HTML_OUT=results/${NOTEBOOK_NAME}_${EXP_CONFIG}.html
LOG=results/${NOTEBOOK_NAME}_${EXP_CONFIG}.log

echo "Starting. $NOTEBOOK -> $NOTEBOOK_OUT; Logfile:" $LOG
time runipy $NOTEBOOK $NOTEBOOK_OUT 2> >(tee $LOG)
echo "Starting. $NOTEBOOK_OUT -> $HTML_OUT"
jupyter nbconvert --to html $NOTEBOOK_OUT
