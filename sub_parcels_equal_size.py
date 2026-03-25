# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 09:23:46 2025

@author: ludo2
"""

import numpy as np
import nibabel as nib 
from sklearn.feature_extraction import image
from sklearn.cluster import AgglomerativeClustering
import time
import os
import shutil
import matplotlib.pyplot as plt
import glob
from sklearn.neighbors import NearestCentroid
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def write_nifti(roi_indices, sub_roi_index, image_template):
    
    im=nib.load(image_template)
    empty_image=np.zeros_like(im.get_fdata(),dtype=int)
    
    for ii in roi_indices:
        empty_image[ii[0],ii[1],ii[2]]=1
        
    nii_=nib.Nifti1Image(empty_image, affine=im.affine, header=im.header)
    
    if sub_roi_index < 10:
        nii_.to_filename('roi_000'+str(sub_roi_index)+'.nii.gz')
    elif sub_roi_index < 100:
        nii_.to_filename('roi_00'+str(sub_roi_index)+'.nii.gz')
    elif sub_roi_index < 1000 :
        nii_.to_filename('roi_0'+str(sub_roi_index)+'.nii.gz')
    else:
        nii_.to_filename('roi_'+str(sub_roi_index)+'.nii.gz')

def check_neigh(roi_coords_in_voxel):
    
    for ii_index,ii in enumerate(roi_coords_in_voxel):
        #print(ii)
        roi_coords_in_voxel_copy=roi_coords_in_voxel.copy()
        roi_coords_in_voxel_copy=np.delete(roi_coords_in_voxel_copy,ii_index,axis=0)
        distances_neigh=(np.abs(ii-roi_coords_in_voxel_copy)>1).sum(axis=1)
        if np.where(distances_neigh>3)[0].shape[0]!=0: # not really convinced
            print(str(ii)+ ' has no direct neigh')

def main():
    
    # Set ROI of interest to left thalamus
    roi_of_int=10
    size_of_int=27
    seg_image=nib.load('MNI152_T1_1mm_seg.nii.gz').get_fdata()
    
    roi_coord=np.vstack(np.where(seg_image==roi_of_int)).T # we used 1022
    
    # if np.round(roi_coord.shape[0]/size_of_int)>(roi_coord.shape[0]/size_of_int):
    #     n_clust=(np.round(roi_coord.shape[0]/size_of_int)-1).astype(int)
    # else:
    #     n_clust=(np.round(roi_coord.shape[0]/size_of_int)).astype(int)
        
    n_clust=int(np.ceil(len(roi_coord)/size_of_int))
    
    seg_image[seg_image!=roi_of_int]=0 # we used 1022
    seg_image[seg_image!=0]=1
    
    conn=image.grid_to_graph(n_x=seg_image.shape[0], n_y=seg_image.shape[1], n_z=seg_image.shape[2],mask=seg_image)

    graph = csr_matrix(conn)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    comp_size=[np.where(labels==ii)[0].size for ii in labels] # from here you can spot voxels that do not belong to the giant component
        
    start_time=time.time()
    labels_clust = AgglomerativeClustering(n_clusters=n_clust, connectivity=conn).fit_predict(roi_coord)
    end_time=time.time()-start_time
    print(end_time)
    
    ### extract_centroids
    clf = NearestCentroid()
    clf.fit(roi_coord, labels_clust)
    centroids=clf.centroids_
    
    # create clusters of even size
    centers = centroids.reshape(-1, 1, roi_coord.shape[-1]).repeat(size_of_int, 1).reshape(-1, roi_coord.shape[-1])
    distance_matrix = cdist(roi_coord, centers)
    start_time=time.time()
    clusters = linear_sum_assignment(distance_matrix)[1]//size_of_int
    end_time=time.time()-start_time
    print(end_time)
    
    unique_labels,counts=np.unique(clusters,return_counts=True)
    
    for ii_index, ii in enumerate(unique_labels):
        dummy=np.where(clusters==ii)[0]
        check_neigh(roi_coord[dummy])
        write_nifti(roi_coord[dummy],ii_index,'MNI152_T1_1mm_seg.nii.gz')
        #print('\n')
    
    os.makedirs(str("ROIs/" + str(roi_of_int)), exist_ok=True)
    files=sorted(glob.glob('roi_*gz'))
    for f in files:
        shutil.move(f, str("ROIs/" + str(roi_of_int)))

if __name__ == '__main__':
    main()  