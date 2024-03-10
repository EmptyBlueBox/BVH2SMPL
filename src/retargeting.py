'''
Remove one spine joint from the preprocessed bvh file, add two hand joints, and change the order of the joints. 
Save it as a .npy array with dimensions (1+24, 3), where 1 represents the offset of the root joint, 
24 represents 24 joints, and 3 represents the x, y, z coordinates for each joint.
'''
import os
import numpy as np
import bvh_utils
from scipy.spatial.transform import Rotation as R

'''
Pelvis(0)
|-- L_Hip(1)
    |-- L_Knee(4)
        |-- L_Ankle(7)
            |-- L_Foot(10)
|-- R_Hip(2)
    |-- R_Knee(5)
        |-- R_Ankle(8)
            |-- R_Foot(11)
|-- Spine1(3)
    |-- Spine2(6)
        |-- Spine3(9)
            |-- Neck(12)
                |-- Head(15)
            |-- L_Collar(13)
                |-- L_Shoulder(16)
                |-- L_Elbow(18)
                    |-- L_Wrist(20)
                        |-- L_Hand(22)
            |-- R_Collar(14)
                |-- R_Shoulder(17)
                |-- R_Elbow(19)
                    |-- R_Wrist(21)
                        |-- R_Hand(23)

Mapping from src bvh skeletal structure to smpl skeletal structure, for example: in the smpl skeletal structure idx==3 is Spine1, 
in the table below we find that idx==3 corresponds to 1, so in the src skeletal structure idx==1 is Spine1 (chest1). 
Additionally, the last two numbers are duplicates, indicating that the src bvh skeleton lacks two hand nodes.
'''
idx_mapping=[0,23,18,1,24,19,2,25,20,3,26,21,5,13,8,6,14,9,15,10,16,11,16,11]

cur_path = os.path.dirname(os.path.abspath(__file__))
bvh_path = cur_path+"/../bvh/"
retargeted_data_path = cur_path+"/../retargeted_data/"
bvh_files=["Armchair1-preprocessed.bvh","Desk1-preprocessed.bvh","Sit54-preprocessed.bvh"]

for i in range(3):
    bvh=bvh_utils.BVHMotion(bvh_path+bvh_files[i])
    retargeted_root_pos = bvh.joint_position[:,0]/100
    retargeted_joint_rot=np.zeros((bvh.motion_length, 24, 3))
    for j in range(bvh.motion_length):
        retargeted_joint_rot[j]=R.from_quat(bvh.joint_rotation[j]).as_euler('XYZ', degrees=False)[idx_mapping]
    retargeted_data=np.concatenate((np.expand_dims(retargeted_root_pos,axis=1),retargeted_joint_rot),axis=1)
    np.save(retargeted_data_path+bvh_files[i].split(".")[0]+"-retargeted.npy",retargeted_data)
    print(retargeted_data[-1])
    print(f"{bvh_files[i].split('.')[0]} retargeted data saved, shape: {retargeted_data.shape}")

    ''' useful for optimization '''
    # bvh.batch_forward_kinematics()
    # bvh.print_dims()
    # retargeted_joint_trans=bvh.joint_translation[:,idx_mapping]
    # print("retargeted_joint_trans.shape: \n", retargeted_joint_trans.shape)
    # np.save(retargeted_data_path+bvh_files[i].split(".")[0]+"-retargeted.npy",retargeted_joint_trans)

    
    