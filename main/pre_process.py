# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 16:32:52 2023

@author: xiao208
"""
import sys
import os
from pathlib import Path
wd = Path().cwd() # working directory
sys.path.insert(1, wd.__str__())
import tifffile
import glob
import os
# import pytiff
import numpy as np
import cv2

from main.custom_filters import kalman_stack_filter
from ScanImageTiffReader import ScanImageTiffReader

def combine_tiff_bruker(fld, preflix):
    folders = [x[0] for x in os.walk(fld) if preflix in x[0]]
    pre_folder = os.path.join(fld, "pre")
    if not os.path.exists(pre_folder):
        os.makedirs(pre_folder)
    for folder in folders:
        basename = os.path.basename(folder)
        fname_Ch2 = basename+'_Cycle00001_Ch2_' # name of the files without numbers
        fls_Ch2 = glob.glob(os.path.join(folders + '//' + preflix,fname_Ch2 + '*.tif'))  #  change tif to the extension you need
        fls_Ch2.sort()  # make sure your files are sorted alphanumerically
    
        m = []
        for file in fls_Ch2:
            m.append(tifffile.imread(file))
        m = np.concatenate(m, axis = 0)
        # m = cm.load_movie_chain(fls_Ch2[0:])
        a = kalman_stack_filter(m).astype(m.dtype)
        a = cv2.normalize(a, None, 0, 2**16-1, cv2.NORM_MINMAX).astype(np.uint16)
        tifffile.imsave(os.path.join(pre_folder,basename + '_combined.tif'),a.astype('uint16'))
        # m1 = movie(a)
        # m1.save(os.path.join(pre_folder,basename + '_combined.tif'))
        # with pytiff.Tiff(os.path.join(pre_folder,basename + '_combined.tif'), "w") as handle:
        #     for i in range(a.shape[0]):
        #         handle.write(a[i,:])
    
    
def combine_tiff(fld, preflix, if_combine = 1):
    files = glob.glob(os.path.join(fld,preflix + '*.tif'))  #  change tif to the extension you need
    pre_folder = os.path.join(fld, "pre")
    if not os.path.exists(pre_folder):
        os.makedirs(pre_folder)
    meta = ScanImageTiffReader(files[0]).metadata()
    # fs = meta.SI.hRoiManager.scanFrameRate
    # dx = 100*meta["RoiGroups"]["imagingRoiGroup"]["scanfields"]["sizeXY"][0]/meta["RoiGroups"]["imagingRoiGroup"]["scanfields"]["pixelResolutionXY"][0]
    # dy = 100*meta["RoiGroups"]["imagingRoiGroup"]["scanfields"]["sizeXY"][1]/meta["RoiGroups"]["imagingRoiGroup"]["scanfields"]["pixelResolutionXY"][1]
    # dxy = (dx, dy)
        # m = []
        # with pytiff.Tiff(file) as handle:
        #   for page in handle.pages:
        #     m.append(page[:])
        # m = np.asarray(m)
    if not if_combine:
        for k, file in enumerate(files):
            print("loading file %d" %k)
            # m = cm.load_movie_chain([file])
            m = ScanImageTiffReader(file).data()
            print("processing file %d"%k)
            a = kalman_stack_filter(m).astype(m.dtype)
            a = cv2.normalize(a, None, 0, 2**16-1, cv2.NORM_MINMAX).astype(np.uint16)
            base_file = os.path.splitext(os.path.basename(file))[0]
            # m1 = movie(a)
            print("saving file %d" %k)
            # m1.save(os.path.join(pre_folder,base_file + '_combined.tif'))
            tifffile.imsave(os.path.join(pre_folder,base_file + '_combined.tif'),a)
            # with pytiff.Tiff(os.path.join(pre_folder,base_file + '_combined.tif'), "w") as handle:
            #     for i in range(a.shape[0]):
            #         handle.write(a[i,:])
    else:
        all_m = []
        print("loading files")
        for k, file in enumerate(files):
        # m = cm.load_movie_chain(files)
            all_m.append(ScanImageTiffReader(file).data())
        m = np.concatenate(all_m, axis = 0)
        print("processing")
        a = kalman_stack_filter(m).astype(m.dtype)
        a = cv2.normalize(a, None, 0, 2**16-1, cv2.NORM_MINMAX).astype(np.uint16)
        # with pytiff.Tiff(os.path.join(pre_folder,preflix + '_combined.tif'), "w") as handle:
        #     for i in range(a.shape[0]):
        #         handle.write(a[i,:])
        # m1 = movie(a)
        print("saving files")
        tifffile.imsave(os.path.join(pre_folder,preflix + '_combined.tif'),a)
        # m1.save(os.path.join(pre_folder,preflix + '_combined.tif'))
    # return fs, dxy


                
def caiman_mc(fld, fs = 30.0, dxy = (0.5,0.5)):
    from caiman import movie
    import caiman as cm
    import sys
    import os
    import glob
    # import pytiff
    from caiman.motion_correction import MotionCorrect
    from caiman.source_extraction.cnmf import params as params

    print("starting motion correction")
    pre_folder = os.path.join(fld, "pre")
    files = glob.glob(os.path.join(pre_folder,'*.tif'))  
    mc_folder = os.path.join(fld, "mc")
    if not os.path.exists(mc_folder):
        os.makedirs(mc_folder)
	# dataset dependent parameters

    fr = fs            # imaging rate in frames per second
    decay_time = 0.5    # length of a typical transient in seconds 
    # dxy = (0.404, 0.404)      # spatial resolution in x and y in (um per pixel)
    # note the lower than usual spatial resolution here
    max_shift_um = (5., 5.)       # maximum shift in um
    patch_motion_um = (20., 20.)  # patch size for non-rigid correction in um
    
    # motion correction parameters
    pw_rigid = True       # flag to select rigid vs pw_rigid motion correction
    # maximum allowed rigid shift in pixels
    max_shifts = [int(a/b) for a, b in zip(max_shift_um, dxy)]
    # start a new patch for pw-rigid motion correction every x pixels
    strides = tuple([int(a/b) for a, b in zip(patch_motion_um, dxy)])
    # overlap between pathes (size of patch in pixels: strides+overlaps)
    overlaps = (5, 5)
    # maximum deviation allowed for patch with respect to rigid shifts
    max_deviation_rigid = 3

    print("starting the cluster")
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)
    
    for k, file in enumerate(files):
        mc_dict = {
            'fnames': file,
            'fr': fr,
            'decay_time': decay_time,
            'dxy': dxy,
            'pw_rigid': pw_rigid,
            'max_shifts': max_shifts,
            'strides': strides,
            'overlaps': overlaps,
            'max_deviation_rigid': max_deviation_rigid,
            'border_nan': 'copy'
        }
    
        opts = params.CNMFParams(params_dict=mc_dict)
           
        print("motion correction for file %d" %k)
        mc = MotionCorrect(file, dview=dview, **opts.get_group('motion'))
        # mc = MotionCorrect(file, **opts.get_group('motion'))
        
        mc.motion_correct(save_movie=True)
        
        m_corr = cm.load(mc.mmap_file)
        print("creating motion corrected tiff for file %d" %k)
        # m_corr.save(os.path.join(os.path.join(mc_folder,os.path.splitext(os.path.basename(file))[0]) + '_corrected.tif'))
        tifffile.imsave(os.path.join(os.path.join(mc_folder,os.path.splitext(os.path.basename(file))[0]) + '_corrected.tif'),m_corr.astype('uint16'))
          		# with pytiff.Tiff(os.path.join(mc_folder,os.path.splitext(os.path.basename(file))[0]), "w") as handle:
           	# 		for i in range(m_corr.shape[0]):
          		# 		handle.write(m_corr[i,:])