from rendering_utils.rotation2xyz import Rotation2xyz
import numpy as np
from trimesh import Trimesh
import os
import torch
from rendering_utils.simplify_loc2rot import joints2smpl
import rendering_utils.rotation_conversions as geometry


class npy2obj:
    def __init__(self, npy_path, device=0, cuda=True):
        self.npy_path = npy_path
        self.motions = np.load(self.npy_path)
        self.rot2xyz = Rotation2xyz(device='cpu')
        self.faces = self.rot2xyz.smpl_model.faces
        self.nframes, self.njoints, self.nfeats = self.motions.shape
        self.opt_cache = {}
        self.num_frames = self.motions.shape[0]
        self.real_num_frames=self.motions.shape[0]
        self.j2s = joints2smpl(num_frames=self.num_frames, device_id=device, cuda=cuda)

        print(f"motion shape: {self.motions.shape}") # (30, 24, 3, 60)
        
        ######################
        ### 1
        # motion_tensor = self.j2s.joint2smpl(self.motions) # [nframes, njoints, 3]
        ### 2
        thetas=torch.tensor(self.motions[:,1:],dtype=float)
        thetas=geometry.matrix_to_rotation_6d(geometry.euler_angles_to_matrix(thetas,'XYZ'))
        root_trans=torch.tensor(self.motions[:,0],dtype=float)
        root_trans=torch.concatenate((root_trans,torch.zeros_like(root_trans,dtype=float)),dim=-1)
        root_trans=torch.unsqueeze(root_trans,axis=1)
        motion_tensor=torch.concatenate((thetas,root_trans),dim=1)
        motion_tensor = motion_tensor.unsqueeze(0).permute(0, 2, 3, 1)
        ######################
        
        print(f"motion tensor shape: {motion_tensor.shape}") # ([1, 25, 6, 60])
        self.motions = motion_tensor.cpu().detach().clone().numpy()

        self.vertices = self.rot2xyz(torch.tensor(self.motions).float(), mask=None,
                                     pose_rep='rot6d', translation=True, glob=True,
                                     jointstype='vertices',
                                     # jointstype='smpl',  # for joint locations
                                     vertstrans=True)

    def get_vertices(self, sample_i, frame_i):
        return self.vertices[sample_i, :, :, frame_i].squeeze().tolist()

    def get_trimesh(self, sample_i, frame_i):
        return Trimesh(vertices=self.get_vertices(sample_i, frame_i),
                       faces=self.faces)

    def save_obj(self, save_path, frame_i):
        mesh = self.get_trimesh(0, frame_i)
        with open(save_path, 'w') as fw:
            mesh.export(fw, 'obj')
        return save_path
    
    # def save_npy(self, save_path):
    #     data_dict = {
    #         'motion': self.motions['motion'][0, :, :, :self.real_num_frames],
    #         'thetas': self.motions['motion'][0, :-1, :, :self.real_num_frames],
    #         'root_translation': self.motions['motion'][0, -1, :3, :self.real_num_frames],
    #         'faces': self.faces,
    #         'vertices': self.vertices[0, :, :, :self.real_num_frames],
    #         'text': self.motions['text'][0],
    #         'length': self.real_num_frames,
    #     }
    #     np.save(save_path, data_dict)
