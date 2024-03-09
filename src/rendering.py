import os
import argparse
from tqdm import tqdm
import rendering_utils.vis_utils

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", type=bool, default=True, help='')
parser.add_argument("--device", type=int, default=0, help='')
params = parser.parse_args()
    
cur_path = os.path.dirname(os.path.abspath(__file__))
retargeted_data_path = cur_path+"/../retargeted_data/"
result_obj_path=cur_path+"/../results-obj/"
if not os.path.exists(result_obj_path):
        os.makedirs(result_obj_path)
retargeted_data=["Armchair1-preprocessed-retargeted.npy","Desk1-preprocessed-retargeted.npy","Sit54-preprocessed-retargeted.npy"]

for i in range(3):
    npy_path=retargeted_data_path+retargeted_data[i]
    result_path=result_obj_path+retargeted_data[i].split("-")[0]
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    npy2obj = rendering_utils.vis_utils.npy2obj(npy_path, device=params.device, cuda=params.cuda)
    for j in tqdm(range(npy2obj.real_num_frames)):
        npy2obj.save_obj(result_path+"/frame_{}.obj".format(j), j)
