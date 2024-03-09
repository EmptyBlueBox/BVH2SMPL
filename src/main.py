import os
import numpy as np
import bvh_utils
from scipy.spatial.transform import Rotation as R

def decompose_rotation_with_yaxis(rotation):
    '''
    输入: rotation 形状为(4,)的ndarray, 四元数旋转
    输出: Ry, Rxz，分别为绕y轴的旋转和转轴在xz平面的旋转，并满足R = Ry * Rxz
    '''
    rot = R.from_quat(rotation)  # 将四元数转换为旋转量
    rot_matrix = rot.as_matrix()  # 计算旋转矩阵
    y_new = (rot_matrix@np.array([0, 1, 0])).T  # 计算y轴旋转后的方向
    y_ori = np.array([0, 1, 0]).T  # y轴原方向
    theta_y = np.arccos(np.dot(y_new, y_ori)/(np.linalg.norm(y_new)*np.linalg.norm(y_ori)))  # 计算y轴旋转角度
    rot_axis_y = np.cross(y_new, y_ori) / np.linalg.norm(np.cross(y_new, y_ori))  # 计算y轴旋转轴
    Ry = R.from_rotvec(theta_y*rot_axis_y)*rot  # 计算y轴旋转量
    return Ry

cur_path= os.path.dirname(os.path.abspath(__file__))
bvh_path= cur_path+"/../bvh/"
bvh_files=["Armchair1.bvh","Desk1.bvh","Sit54.bvh"]
crop_range=[(380,680),(60,360),(0,300)]
for i in range(3):
    bvh=bvh_utils.BVHMotion(bvh_path+bvh_files[i])
    '''截取指定帧数的动作'''
    bvh=bvh.sub_sequence(start=crop_range[i][0],end=crop_range[i][1])
    '''让每段动作的末尾根关节都在原点'''
    bvh.joint_position-=bvh.joint_position[-1] # 末尾根关节在原点
    '''让每段动作的末尾根关节都朝向 x 轴正方向'''
    end_frame_root_rotation_y=decompose_rotation_with_yaxis(bvh.joint_rotation[-1,0])
    bvh.joint_rotation[:,0]=(end_frame_root_rotation_y.inv()*R.from_quat(bvh.joint_rotation[:,0])).as_quat()
    bvh.joint_position[:,0]=end_frame_root_rotation_y.inv().apply(bvh.joint_position[:,0])
    bvh.print_dims()
    bvh.put_back_bvh("preprocessed")