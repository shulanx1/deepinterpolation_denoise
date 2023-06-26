# deepinterpolation_denoise #


 Auto-combine and denoise the 2p Ca imaging .tiff files using deepinterpolation (Lecoq, J., Oliver, M., Siegle, J.H. et al. Removing independent noise in systems neuroscience data using DeepInterpolation. Nat Methods 18, 1401–1408 (2021). https://doi.org/10.1038/s41592-021-01285-2)


## Installation ##
* Download and install anaconda (https://www.anaconda.com)
* Install deepinterpolation https://github.com/AllenInstitute/deepinterpolation.
Open anaconda prompt, on the command line run: <br />
 -- conda create -n deepinterpolation python=3.7<br />
Download the deepinterpolation repository<br />
 -- git clone https://github.com/AllenInstitute/deepinterpolation.git<br />
Go to the directory just downloaded<br />
 -- cd deepinterpolation<br />
Activate the environment<br />
-- conda activate deepinterpolation<br />
Install<br />
-- pip install -r requirements.txt<br />
-- python setup.py install<br />

Note that this is for use with CPU ONLY! If you're working with GPU, follow the installing instruction in https://github.com/AllenInstitute/deepinterpolation
* Install other dependencies
-- pip install scanimage-tiff-reader tiffile opencv-python<br />

* Install jupyter notebook<br />
-- conda install -c conda-forge notebook nb_conda_kernels ipywidgets <br />

## How to run ##
* Open jupyter notebook<br />
	open anaconda prompt, move to the local directory where the folder with pipeline.ipynb is saved<br />
		- cd \path\to\deepinterpolation_denoise<br />
		- jupyter notebook
* Modify the pipeline files<br />
	on the notebook, in the second cell, change "fld" to where the videos to be processed is saved, and "preflix" to the base name of the videos (without the indexing)<br />
* Run the pipeline cell by cell	
* Exit the environment after finished
 -- conda deactivate
	

## Files ##

* //main: functions for combining tiff files, pre-filtering and denoising
* //demo_dataset: test datasets
* //model_folder: pre-trained models(github: https://github.com/AllenInstitute/deepinterpolation)
* //pipeline: demo pipeline to denoise the datasets in the demo_dataset folder