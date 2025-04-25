#!/bin/bash

# rs_qsub.sh name_queue 1:00:00 bash path/to/this/launch_queue.sh /path/to/python_script.py

# Check if script argument is provided
if [ -z "$1" ]; then
  echo "Please provide the full path to the Python script."
  exit 1
fi

# Extract the script name from the full path
script_name=$(basename "$1")
script_dir=$(dirname "$1")
project_name=nm_gnn

# Attach to Docker container | UPDATE docker path
# NOTE: CHANGE THE DOCKER CONTAINER NAME ACCORDING TO YOUR WORKING CONTAINER (i.e., lightning_templ_$USER)
docker restart ${project_name}_${USER}_24.05
docker exec ${project_name}_${USER}_24.05 bash -c "cd $script_dir && python $script_name --conf config_files/${script_name%.py}.yaml"