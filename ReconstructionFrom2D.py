import sys
import numpy as np
import os
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
from scipy import io
from glob import glob
import json
import pandas as pd

from fundamental_matrix_estimation import *
from sfm_utils import *

from factorization_method import *
from triangulation import *

def plotting(X, Y, Z):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.gca(projection='3d')
    for i in range(len(X)):
        ax.scatter(X[i], Y[i], Z[i], c='k', depthshade=True, s=2)
        ax.text(X[i] + 1, Y[i] + 1, Z[i] + 1, s=str(i))

    # ax.scatter(X, Y, Z, c='k', depthshade=True, s=2)

    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0

    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()


if __name__ == '__main__':
    images_data_dir = 'data_MDIL/images/'
    pose2d_data_dir = 'data_MDIL/pose2d/'

    images_jpg_list = glob(images_data_dir + '*.jpg')
    im0 = imread(images_jpg_list[0])
    im_height, im_width, _ = im0.shape

    pose_json_list = glob(pose2d_data_dir+'*.json')
    N_views = len(pose_json_list)
    print(f"# of views : {N_views}")

    # Parsing 18 camera parameters
    cam_param_path = r"./camera_parameters.mat"
    cam_param = io.loadmat(cam_param_path)
    K = cam_param['K']
    R = cam_param['R']
    T = cam_param['T']
    print(cam_param['K']) # 18x3x3
    print(cam_param['R']) # 18x3x3
    print(cam_param['T']) # 18x3


    # Parsing 18 json files.
    for i, pose in enumerate(pose_json_list):
        with open(pose) as json_file:
            pose_data = json.load(json_file)
            pose_keypoints_2D = pose_data['people'][0]['pose_keypoints_2d']
            pose_keypoints_2D_pairs_matrix = np.zeros((0,3))

            for j, keypoint in enumerate(pose_keypoints_2D):
                matrix_index = j % 3
                if matrix_index == 0:
                    pose_keypoints_2D_matrix = np.zeros((1, 3), dtype='f')
                    pose_keypoints_2D_matrix[0,matrix_index] = keypoint
                elif matrix_index == 1:
                    pose_keypoints_2D_matrix[0, matrix_index] = keypoint
                elif matrix_index == 2:
                    pose_keypoints_2D_matrix[0, matrix_index] = keypoint
                    pose_keypoints_2D_pairs_matrix = np.vstack([pose_keypoints_2D_pairs_matrix, pose_keypoints_2D_matrix])
            print('-' * 10 + pose + '-'*10)
            print(pose_keypoints_2D_pairs_matrix)

            if i==0:
                all_pose_keypoints_2D_pairs_matrix = np.empty((pose_keypoints_2D_pairs_matrix.shape[0], pose_keypoints_2D_pairs_matrix.shape[1], N_views))
            all_pose_keypoints_2D_pairs_matrix[:,:,i] = pose_keypoints_2D_pairs_matrix

    # 'all_pose_keypoints_2D_pairs_matrix' is a 25x3x18 array.
    # print(all_pose_keypoints_2D_pairs_matrix)

    # Make camera matrices
    cam_matrices = np.zeros((N_views,3,4))
    for i in range(N_views-1):
        RT = np.hstack((R[i],np.reshape(T[i],(3,1))))
        cam_matrices[i,:,:] = K[i].dot(RT)


    # Find 3D points of each keypoints
    object_3D = np.zeros((25,3))
    for i, key in enumerate(all_pose_keypoints_2D_pairs_matrix):
        x_array = key[0,:]
        y_array = key[1,:]
        xy_array = np.column_stack((x_array,y_array)) # 18x2 matrix

        DF_pairs = pd.DataFrame(xy_array)
        DF_pairs = DF_pairs[((DF_pairs[0] != 0.0) & (DF_pairs[1] != 0.0))]
        #DF_pairs = DF_pairs.drop(columns=[0,2], axis=1)
        xy_array = pd.DataFrame.to_numpy(DF_pairs)
        alive_index_list = DF_pairs.index.to_list()

        object_point = linear_estimate_3d_point(xy_array, cam_matrices) # Nx2, Nx3x4
        object_3D[i] = object_point

        # linear estimate
        linear_estimated_pair_object_point = np.zeros((xy_array.shape[0],3)) #18x3
        pairs = np.zeros((len(alive_index_list),4))
        for j in range(xy_array.shape[0]):
            if j == xy_array.shape[0]-1:
                pair = np.vstack((xy_array[j], xy_array[0]))  # 2x2
                pair_cam_matrices = np.zeros((2, 3, 4))
                pair_cam_matrices[0, :, :] = cam_matrices[j]
                pair_cam_matrices[1, :, :] = cam_matrices[0]
            else:
                pair = np.vstack((xy_array[j],xy_array[j+1])) # 2x2
                pair_cam_matrices = np.zeros((2,3,4))
                pair_cam_matrices[0,:,:] = cam_matrices[j]
                pair_cam_matrices[1,:,:] = cam_matrices[j+1]

            pairs[j,:] = pair.reshape(1,-1)
            # object_point = linear_estimate_3d_point(pairs, pairs_cam_matrices) # Nx2, Nx3x4
            # linear_estimated_pair_object_point[j] = object_point
            # object_3D[i] = object_point

        #
        # print(alive_index_list)
        # print(pairs)
        #
        # # bundle adjustment to linear_estimated_pair_object_point
        # final_estimated_object_point = np.zeros((1,3))
        # # blah blah
        # focal_length = 1600
        # frames = [0] * (len(alive_index_list))
        #
        # # for j in range(3):
        # for j in range(len(alive_index_list)):
        #     if j == len(alive_index_list)-1:
        #         frames[j] = Frame(pairs[j, :].reshape(1, -1), focal_length, im_width, im_height,
        #                           K[alive_index_list[j], :, :], K[alive_index_list[0], :, :],
        #                           np.linalg.inv(R[alive_index_list[j], :, :]).dot(R[alive_index_list[0], :, :]),
        #                           np.reshape(T[alive_index_list[0]] - T[alive_index_list[j]], (3, 1)))
        #     else:
        #         frames[j] = Frame(pairs[j, :].reshape(1, -1), focal_length, im_width, im_height,
        #                           K[alive_index_list[j], :, :], K[alive_index_list[j + 1], :, :],
        #                           np.linalg.inv(R[alive_index_list[j], :, :]).dot(R[alive_index_list[j + 1], :, :]),
        #                           np.reshape(T[alive_index_list[j + 1]] - T[alive_index_list[j]], (3, 1)))
        #
        # print(f"Merge {i}-th keypoint")
        # merged_frame = merge_all_frames(frames)
        #
        # final_estimated_object_point=merged_frame.structure[0,:]
        # print(merged_frame.structure)
        #
        # object_3D[i] = final_estimated_object_point
    plotting(object_3D[:,0],object_3D[:,1],object_3D[:,2])

    print(object_3D)
    print('=========END=========')


