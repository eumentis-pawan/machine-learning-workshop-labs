#!/bin/bash

# This script will split out a directory of files into other random directories, namely for splitting an image set into a training/testing/val set of folders

# Usage: ./sort_photos.sh ./flowers/daisy/ ./flowers_sorted/daisy/test 25
# Usage: ./sort_photos.sh ./flowers/daisy/ ./flowers_sorted/daisy/val 5

SRC_DIRECTORY=$1
TARGET_DIRECTORY=$2
MOVE_PERCENTAGE=$3

TOTAL_FILES=$(ls ${SRC_DIRECTORY} | wc -l)
MOVE_NUM=$(bc <<<"${TOTAL_FILES}*${MOVE_PERCENTAGE}/100")

CURRENT_MOVE_NUM="0"

echo "Total files in ${SRC_DIRECTORY}: ${TOTAL_FILES}"
echo "${MOVE_PERCENTAGE}% Targeted to move to ${TARGET_DIRECTORY}: ${MOVE_NUM}"

ls ${SRC_DIRECTORY} | sort -R | while read file; do
  if [[ $CURRENT_MOVE_NUM -lt $MOVE_NUM ]]; then
    echo "[$((CURRENT_MOVE_NUM + 1))/$MOVE_NUM] Moving ${SRC_DIRECTORY}${file} to ${TARGET_DIRECTORY}"
    mv ${SRC_DIRECTORY}${file} $TARGET_DIRECTORY
    CURRENT_MOVE_NUM=$(( $CURRENT_MOVE_NUM + 1 ))
  fi
done