#!/bin/sh
echo "<< Start pipeline.sh >>"
echo " "
python data_creation.py
python data_preprocessing.py
python model_preparation.py
python model_testing.py
echo " "
echo "<< Finish pipeline.sh >>"
