# deepinterpolation_denoise
 
# README #

* Auto-combine and denoise the 2p Ca imaging .tiff files using deepinterpolation (Lecoq, J., Oliver, M., Siegle, J.H. et al. Removing independent noise in systems neuroscience data using DeepInterpolation. Nat Methods 18, 1401–1408 (2021). https://doi.org/10.1038/s41592-021-01285-2)


### Installation ###
* Download and install anaconda (https://www.anaconda.com)
* Install deepinterpolation https://github.com/AllenInstitute/deepinterpolation
	open anaconda prompt, on the command line run: 
		- conda create -n deepinterpolation python=3.7
		- pip install deepinterpolation
	Note that this is for use with CPU ONLY! If you're working with GPU, follow the installing instruction in https://github.com/AllenInstitute/deepinterpolation
* Install other dependencies
	open anaconda prompt, activate the environment on the command line:
		- conda activate deepinterpolation
	install ScanImageTiffReader
		- pip install scanimage-tiff-reader


### How to run ###
* Open jupyter notebook
	open anaconda prompt, move to the local directory where the folder with pipeline.ipynb is saved
		- cd \path\to\deepinterpolation_denoise
		- jupyter notebook
* Modify the pipeline files
	on the notebook, in the second cell, change "fld" to where the videos to be processed is saved, and "preflix" to the base name of the videos (without the indexing)
* Run the pipeline cell by cell	
	

### Files ###

* //main: functions for combining tiff files, pre-filtering and denoising
* //demo_dataset: test datasets
* //model_folder: pre-trained models(github: https://github.com/AllenInstitute/deepinterpolation)
* //pipeline: demo pipeline to denoise the datasets in the demo_dataset folder