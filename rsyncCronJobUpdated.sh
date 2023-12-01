#!/bin/bash

# This script is supposed to run only once to add a cron scheduled task
# that run daily to sync graphDB file path with Synology file path that store
# latest PASCAL data.

#info needed: path for the two dir; path for preprocessing.pye

# Default source and destination directories
DEFAULT_SOURCE_DIR="$HOME/test/source" # placeholder 
DEFAULT_DEST_DIR="http://localhost:7474/db/data/transaction/commit" # not sure if this is the correct url

# Prompt the user for the source directory, with the default as a suggestion
read -p "Enter the source directory (default: $DEFAULT_SOURCE_DIR): " SOURCE_DIR
SOURCE_DIR=${SOURCE_DIR:-$DEFAULT_SOURCE_DIR}

# Prompt the user for the destination directory, with the default as a suggestion
read -p "Enter the destination directory (default: $DEFAULT_DEST_DIR): " DEST_DIR
DEST_DIR=${DEST_DIR:-$DEFAULT_DEST_DIR}

# Prompt the user for the Neo4j username and password
read -p "Enter your Neo4j username: " USER
read -sp "Enter your Neo4j password: " PASS
echo

# Loop through each image in the directory
for IMAGE in ${SOURCE_DIR}/*.jpg 
do 
    # Extract the filename from the path
    FILENAME=$(basename ${IMAGE})

    # Cypher query to create a node for the image
    QUERY="MERGE (n:Image {filename: '${FILENAME}'})"

    # Convert query to JSON format
    JSON="{\"statements\" : [ {\"statement\" : \"${QUERY}\"} ]}"
    
PYTHON_FILE_DIR="$HOME/placeholder/preprocessing.py" #Need path to preprocess.py

# Upload data
curl -H 'Content-Type: application/json' -H 'Accept: application/json; charset=UTF-8' -u ${USER}:${PASS} -d "${JSON}" ${DEFAULT_DEST_DIR}
done
# Define the rsync command
RSYNC_COMMAND="rsync -avr --ignore-existing $SOURCE_DIR/ $DEST_DIR"

# Add the rsync command as a cron job (adjust the timing as needed)
(crontab -l ; echo "0 1 * * * $RSYNC_COMMAND ; python3 $PYTHON_FILE_DIR") | crontab -


