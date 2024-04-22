#!/bin/sh
echo "<< Start pipeline.sh >>"
echo " "
python data_creation.py $1
python model_preprocessing.py $1
python model_preparation.py $1
python model_testing.py $1
echo " "
echo "<< Finish pipeline.sh >>"
