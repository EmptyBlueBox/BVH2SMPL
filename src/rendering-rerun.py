import smplx
import argparse
import numpy as np
import torch
import rerun as rr


def setup_device():
    """设置计算设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return device


device = setup_device()
max_frame_num = 10  # 使用的帧数


def compute_vertex_normals(vertices, faces):
    """
    使用向量化操作计算顶点法向量。
    """
    # 获取三角形的顶点
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # 计算每个面的法向量
    normals = np.cross(v1 - v0, v2 - v0)
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]

    # 将法向量累加到顶点上
    vertex_normals = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], normals)

    # 归一化顶点法向量
    norms = np.linalg.norm(vertex_normals, axis=1)
    vertex_normals = (vertex_normals.T / norms).T

    return vertex_normals


def load_human_mesh(pose_path, orient_path, transl_path):

    SMPL_MODEL_PATH = '/home/***/Documents/SMPLX/models'  # SMPLX 模型路径
    human_model = smplx.create(model_path=SMPL_MODEL_PATH,
                               model_type='smplx',
                               gender='neutral',
                               use_face_contour=False,
                               num_betas=10,
                               num_expression_coeffs=10,
                               ext='npz',
                               batch_size=max_frame_num)

    smpl_params = {
        'poses': np.load(pose_path),
        'orientation': np.load(orient_path),
        'translation': np.load(transl_path)
    }

    # 后续处理
    smpl_params['poses'] = torch.tensor(smpl_params['poses'].reshape(-1, 21, 3), dtype=torch.float32)
    smpl_params['orientation'] = torch.tensor(smpl_params['orientation'].reshape(-1, 3), dtype=torch.float32)
    smpl_params['translation'] = torch.tensor(smpl_params['translation'].reshape(-1, 3), dtype=torch.float32)

    print(f'smpl_params.pose:{smpl_params["poses"].shape}')
    print(f'smpl_params.orientation:{smpl_params["orientation"].shape}')
    print(f'smpl_params.translation:{smpl_params["translation"].shape}')

    output = human_model(body_pose=smpl_params['poses'],
                         global_orient=smpl_params['orientation'],
                         transl=smpl_params['translation'])
    vertices = output.vertices.detach().cpu().numpy()
    faces = human_model.faces

    # 计算顶点法向量
    vertex_normals = np.zeros_like(vertices)
    for i in range(vertices.shape[0]):
        vertex_normals[i] = compute_vertex_normals(vertices[i], faces)

    print(f'vertices.shape:{vertices.shape}')
    print(f'body_model.faces.shape:{human_model.faces.shape}')
    print(f'vertex_normals.shape:{vertex_normals.shape}')

    human_mesh = {'vertices': vertices, 'faces': faces, 'vertex_normals': vertex_normals}

    return human_mesh


def load_human():
    human = []
    human_mesh = load_human_mesh()
    human.append(human_mesh)
    return human


def write_rerun(human: list):
    parser = argparse.ArgumentParser(description="Logs rich data using the Rerun SDK.")
    rr.script_add_args(parser)
    args = parser.parse_args()
    rr.script_setup(args, 'SMPL-X')
    rr.set_time_seconds("stable_time", 0)

    frames_per_second = 30
    for i in range(max_frame_num):  # 实际的帧数
        time = i / frames_per_second
        rr.set_time_seconds("stable_time", time)

        rr.log(
            'human',
            rr.Mesh3D(
                vertex_positions=human[0]['vertices'][i],
                triangle_indices=human[0]['faces'],
                vertex_normals=human[0]['vertex_normals'][i],
            ),
        )

    rr.script_teardown(args)


def main():
    human = load_human()
    write_rerun(human=human)


if __name__ == '__main__':
    main()
