# MILimbEEG-Classsifier

This project uses the MILimbEEG dataset, which is an EEG dataset for motor and motor imagery tasks recorded for 7 different tasks. link for the dataset is https://data.mendeley.com/datasets/x8psbz3f6x/2.

Prerequisites:
*pip install zipfile36
*pip install scipy
*pip install os

The code:
Three prepatory codes have to run before the model compilation. These are to be executed in order:
1) download_and_prepare: Downloads and unzips the dataset and structures the folder to prepare it for preprocessing. (requires internet access)
2) bandpass: A required fitering that passes the data through a filter for frequencies of band 1.0Hz to 50Hz. It also preserves and negative values and the output of this code needs to be used to process in the ML mdoel.
3) encoder: creates a metadata excel sheet that decodes the naming scheme of the excel sheets for ease of access during the training process. it also creates a column that points to the local url of each of the excel sheets and clearly marks the output labels.
