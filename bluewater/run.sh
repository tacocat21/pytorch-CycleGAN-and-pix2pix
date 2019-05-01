#!/bin/bash
# This script runs multiple batch jobs to test different hyperparameter
# settings. For each setting, it creates a different PBS file and calls
# it.

# MODIFY THESE
declare training_file="train.py"
declare walltime="05:00:00"
declare jobname="hw4_transfer_learning"
declare netid="tyamamo2"
declare directory="~/scratch/ie534/hw4"

# Declare the hyperparameters you want to iterate over
declare -a trial_number=(0 1 2 3 4 5 6 7)

# For each parameter setting we generate a new PBS file and run it
for trial in "${trial_number[@]}"
do
  python ../generate_pbs.py $training_file $walltime $jobname $netid $directory $trial > run.pbs
  echo "Submitting $trial"
  qsub run.pbs -A bauh
done
