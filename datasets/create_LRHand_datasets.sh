#!/bin/bash

current_folder=/home/yjy/PycharmProjects/Finger_root_Eager/datasets

image_folder=$current_folder/LHand/palmprint_trainval
coordinates_filename_folder=$current_folder/LHand/palmprint_trainval
list_folder=$current_folder/LHand
output_dir=$current_folder/LHand/tfrecord

python build_pp_data.py \
	--image_folder=$image_folder \
	--coordinates_filename_folder=$coordinates_filename_folder \
	--list_folder=$list_folder \
	--output_dir=$output_dir 

image_folder=$current_folder/RHand/palmprint_trainval
coordinates_filename_folder=$current_folder/RHand/palmprint_trainval
list_folder=$current_folder/RHand
output_dir=$current_folder/RHand/tfrecord

python build_pp_data.py \
	--image_folder=$image_folder \
	--coordinates_filename_folder=$coordinates_filename_folder \
	--list_folder=$list_folder \
	--output_dir=$output_dir 
