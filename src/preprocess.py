import os
import numpy as np
import bvh_utils
from scipy.spatial.transform import Rotation as R

def decompose_rotation_with_yaxis(rotation):
    """
    This Python function decomposes a quaternion rotation into a rotation around the y-axis and a
    rotation in the xz plane with the y-axis as the rotation axis.
    
    @param rotation rotation is a quaternion representing a 3D rotation.
    
    @return The function `decompose_rotation_with_yaxis` returns the rotation `Ry` around the y-axis and
    the rotation around the axis in the xz plane, such that the original rotation `R` can be decomposed
    as `R = Ry * Rxz`.
    """
    rot = R.from_quat(rotation)  # Convert quaternion to rotation amount
    rot_matrix = rot.as_matrix()  # Calculate the rotation matrix
    y_new = (rot_matrix@np.array([0, 1, 0])).T  # Calculate the direction after rotation around the y-axis.
    y_ori = np.array([0, 1, 0]).T  # y-axis original direction
    theta_y = np.arccos(np.dot(y_new, y_ori)/(np.linalg.norm(y_new)*np.linalg.norm(y_ori)))  # Calculate the y-axis rotation angle
    rot_axis_y = np.cross(y_new, y_ori) / np.linalg.norm(np.cross(y_new, y_ori))  # Calculate the y-axis rotation axis
    Ry = R.from_rotvec(theta_y*rot_axis_y)*rot  # Calculate the y-axis rotation amount
    return Ry

cur_path= os.path.dirname(os.path.abspath(__file__))
bvh_path= cur_path+"/../bvh/"
bvh_files=["Armchair1.bvh","Desk1.bvh","Sit54.bvh"]
crop_range=[(480,680),(160,360),(100,300)]
for i in range(3):
    bvh=bvh_utils.BVHMotion(bvh_path+bvh_files[i])
    '''Capture the specified number of frames of action'''
    bvh=bvh.sub_sequence(start=crop_range[i][0],end=crop_range[i][1])
    '''Ensure the end joint of each action segment is at the origin.'''
    bvh.joint_position-=bvh.joint_position[-1] # 末尾根关节在原点
    '''Ensure the end joint of each action segment is oriented towards the positive direction of the x-axis.'''
    end_frame_root_rotation_y=decompose_rotation_with_yaxis(bvh.joint_rotation[-1,0])
    bvh.joint_rotation[:,0]=(end_frame_root_rotation_y.inv()*R.from_quat(bvh.joint_rotation[:,0])).as_quat()
    bvh.joint_position[:,0]=end_frame_root_rotation_y.inv().apply(bvh.joint_position[:,0])
    bvh.print_dims()
    bvh.put_back_bvh("preprocessed")