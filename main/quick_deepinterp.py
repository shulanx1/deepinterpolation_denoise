# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 18:54:54 2023

@author: xiao208
"""
from deepinterpolation.cli.fine_tuning import FineTuning
from deepinterpolation.cli.inference import Inference
import os
import glob
import datetime
import h5py
import sys
import shutil
import shlex
import subprocess
# import pytiff
# import caiman as cm
# from caiman import movie
from pathlib import Path
wd = Path().cwd() # working directory
sys.path.insert(1, wd.__str__())


def quick_finetuning(fld, movie_idx = 0, model_name = '2021_07_31_09_31_03_528689_unet_1024_search_mean_squared_error_pre_30_post_30_feat_32_power_2_depth_4_unet_True-0100-0.5733.h5'):
    finetuning_params = {}
    generator_param = {}
    generator_test_param = {}
    nb_frame_training = 500
    input_model_path = os.path.join(wd, "model_folder", model_name)
    input_movies = glob.glob(os.path.join(fld, "pre" , "*.tif")) 
    input_movie_path = input_movies[movie_idx]
    
    # copy model file to local directory
    local_model_fld = os.path.join(fld, "model_folder")
    local_model_path = os.path.join(local_model_fld, model_name)
    if not os.path.exists(local_model_fld):
        os.makedirs(local_model_fld)
    if not os.path.exists(local_model_path):
        shutil.copyfile(input_model_path, local_model_path)

    generator_param["name"] = "SingleTifGenerator"  # Name of object (use SingleTifGenerator for tiff files)
    generator_param["pre_frame"] = 30
    generator_param["post_frame"] = 30
    generator_param["data_path"] = input_movie_path
    generator_param["batch_size"] = 1 # This is small because Colab GPUs do have very smaller memory. Increase on better cards. 
    generator_param["start_frame"] = 0
    generator_param["end_frame"] = -1
    generator_param["total_samples"] = nb_frame_training
    generator_param["pre_post_omission"] = 0  # Number of frame omitted before and after the predicted frame
    
    generator_test_param["name"] = "SingleTifGenerator"  # Name of object (use SingleTifGenerator for single tiff files or MultiContinuousTifGenerator for an ordered serie of Tiffs)
    generator_test_param["pre_frame"] = 30
    generator_test_param["post_frame"] = 30
    generator_test_param["data_path"] = input_movie_path
    generator_test_param["batch_size"] = 1
    generator_test_param["start_frame"] = 0
    generator_test_param["end_frame"] = -1
    generator_test_param["total_samples"] = 100  # This is use to measure validation loss
    generator_test_param["pre_post_omission"] = 0  # Number of frame omitted before and after the predicted frame

    # Those are parameters used for the training process
    finetuning_params["name"] = "transfer_trainer"
    
    # Change this path to any model you wish to improve
    finetuning_params["model_source"] = {
      "local_path": local_model_path
    }
    
    # An epoch is defined as the number of batches pulled from the dataset before measuring validation loss.
    # It is mostly for performance tracking 
    # Because our datasets are VERY large. Often, we cannot
    # go through the entirety of the data so we define an epoch
    # slightly differently than is usual.
    steps_per_epoch = 200
    finetuning_params["steps_per_epoch"] = steps_per_epoch
    finetuning_params[
    "period_save"
    ] = 25
    # network model is potentially saved during training between a regular
    # nb of epochs. Useful to go back to models during training
    
    finetuning_params["learning_rate"] = 0.0001
    finetuning_params["loss"] = "mean_squared_error"
    finetuning_params["output_dir"] = local_model_fld
        
    finetuning_params["use_multiprocessing"] = False
    finetuning_params["caching_validation"] = False
    
    args = {
    "finetuning_params": finetuning_params,
    "generator_params": generator_param,
    "test_generator_params": generator_test_param,
    "output_full_args": True
    }
    
    finetuning_obj = FineTuning(input_data=args, args=[])
    
    print("Starting fine-tuning")
    
    finetuning_obj.run()
    
    print("Fine-tuning finished")
    
    return generator_param
    
def quick_inference(fld, generator_param):
    input_movies = glob.glob(os.path.join(fld, "pre" , "*.tif")) 
    output_dir = os.path.join(fld, "denoise")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    inference_param = {}
    # This is the name of the underlying inference class called
    inference_param["name"] = "core_inferrence"
    
    # Where the output of the previous training is stored
    local_path =  glob.glob(os.path.join(fld, "model_folder", "*_mean_squared_error_transfer_model.h5"))
    local_path.sort()
    
    inference_param["model_source"] = {
    "local_path": local_path[-1]
    }

    # This option is to add blank frames at the onset and end of the output
    # movie if some output frames are missing input frames to go through
    # the model. This could be present at the start and end of the movie.
    inference_param["output_padding"] = True
    
    # this is an optional parameter to bring back output data to a given
    # precision. Read the CLI documentation for more details.
    # this is available through
    # 'python -m deepinterpolation.cli.inference --help'
    inference_param["output_datatype"] = 'uint16'
    # inference_param["use_multiprocessing"] = False

    from ScanImageTiffReader import ScanImageTiffReader
    
    for k, file in enumerate(input_movies):
        print("Preparing data for inference file %d" %k)
        # Initialize meta-parameters objects
        # with pytiff.Tiff(file) as handle:
        #     n_frame = len(handle.pages)
        # m = cm.load_movie_chain([file])
        
        m = ScanImageTiffReader(file).data()
        n_frame = m.shape[0]
        # We are reusing the data generator for training here.
        generator_param["start_frame"] = 0
        generator_param["end_frame"] = -1
        generator_param["total_samples"] = n_frame
        # generator_param["normalize_cache"] = True
        
    
        base_file = os.path.splitext(os.path.basename(file))[0]
        
        unique_time = str(datetime.datetime.now()).replace(".","-").replace(":","-").replace(" ","-")
        
        # Replace this path to where you want to store your output file
        inference_param["output_file"] = os.path.join(output_dir, base_file+'_denoise.h5')
    
        
        args = {
        "generator_params": generator_param,
        "inference_params": inference_param,
        "output_full_args": True
        }
        
        
        
        print("Starting inference for file %d" %k)
        inference_obj = Inference(input_data=args, args=[])
        inference_obj.run()
        
        # command = "C:\\ProgramData\\Anaconda3\\envs\\deepinterpolation\\python.exe"
        # inference_subprocess = subprocess.Popen([command ,os.path.join(Path().cwd(), "main", "deepinterp_inference.py"),'args'], shell = True)
        # inference_subprocess.wait()
        
        print("Inference finished for file %d" %k)
        
        print("generate tiff for file %d" %k)
        h5_2_tiff(inference_param["output_file"], file, generator_param)
        print("tiff created for file %d" %k)
        
def h5_2_tiff(out_file, orig_file, generator_param):
    import numpy as np
    import tifffile
#    import pytiff
    import cv2
    
    data = np.asarray(h5py.File(out_file,'r')['data'])
    data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # orig_data = []
    # with pytiff.Tiff(orig_file, 'r') as handle:
    #     for page in handle.pages:
    #         orig_data.append(np.asarray(page[:]))
    # orig_data = np.asarray(orig_data)
    from ScanImageTiffReader import ScanImageTiffReader
    orig_data = ScanImageTiffReader(orig_file).data()
    orig_data = cv2.normalize(orig_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    data[:generator_param["pre_frame"],:] = orig_data[:generator_param["pre_frame"],:]
    data[-generator_param["post_frame"]:,:] = orig_data[-generator_param["post_frame"]:,:]
    basename = os.path.splitext(os.path.basename(out_file))[0]
    final_dir = os.path.join(os.path.dirname(os.path.dirname(out_file)), "final")
    if not os.path.exists(final_dir):
        os.makedirs(final_dir)
    tifffile.imsave(os.path.join(final_dir, basename) + '.tif',data)
    # m1 = movie(data)
    # m1.save(os.path.splitext(out_file)[0] + '.tif')
    
    # with pytiff.Tiff(os.path.splitext(out_file)[0] + '.tif', "w") as handle:
    #   for i in range(data.shape[0]):  
    #     if (i < generator_param["pre_frame"]) or (i > data.shape[0]-generator_param["post_frame"]):
    #         handle.write(orig_data[i,:])
    #     else:
    #         handle.write(data[i,:])
        