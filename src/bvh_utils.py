import numpy as np
import copy
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


'''
In the comments, N uniformly represents the number of frames, and M represents the number of joints. 
"position, rotation" indicate local translation and rotation
"translation, orientation" indicate global translation and rotation.
'''


class BVHMotion():
    def __init__(self, bvh_file_path=None) -> None:
        self.bvh_file_path = bvh_file_path
        self.bvh_file_name = bvh_file_path.split('/')[-1].split('.')[0]
        self.euler='YXZ'

        # 一些 meta data
        self.original_meta_data = []

        self.joint_name = []
        self.end_sites = []
        self.joint_channel = []
        self.joint_parent = []

        # 一些local数据, 对应bvh里的channel, XYZposition和 XYZrotation
        #! 这里我们把没有XYZ position的joint的position设置为offset, 从而进行统一
        self.joint_position = None  # (N,M,3) 的ndarray, 局部平移
        self.joint_rotation = None  # (N,M,4)的ndarray, 用四元数表示的局部旋转
        self.joint_translation = None  # (N,M,3)的ndarray, 全局平移
        self.joint_orientation = None  # (N,M,4)的ndarray, 用四元数表示的全局旋转

        # 骨骼中点的变化比例
        self.marker_fluctuation_rate = None

        if bvh_file_path is not None:
            self.load_motion(bvh_file_path)

        # self.print_dims()
        pass

    def load_meta_data(self, bvh_path):
        with open(bvh_path, 'r') as f:
            channels = []
            joints = []
            joint_parents = []
            joint_offsets = []
            end_sites = []
            parent_stack = [None]
            original_meta_data = []
            for line in f:
                original_meta_data.append(line)

                if 'ROOT' in line or 'JOINT' in line:
                    joints.append(line.split()[-1])
                    joint_parents.append(parent_stack[-1])
                    channels.append('')
                    joint_offsets.append([0, 0, 0])

                elif 'End Site' in line:
                    end_sites.append(len(joints))
                    joints.append(parent_stack[-1] + '_end')
                    joint_parents.append(parent_stack[-1])
                    channels.append('')
                    joint_offsets.append([0, 0, 0])

                elif '{' in line:
                    parent_stack.append(joints[-1])

                elif '}' in line:
                    parent_stack.pop()

                elif 'OFFSET' in line:
                    joint_offsets[-1] = np.array([float(x)
                                                  for x in line.split()[-3:]]).reshape(1, 3)

                elif 'CHANNELS' in line:
                    trans_order = []
                    rot_order = []
                    for token in line.split():
                        if 'position' in token:
                            trans_order.append(token[0])

                        if 'rotation' in token:
                            rot_order.append(token[0])

                    channels[-1] = ''.join(trans_order) + ''.join(rot_order)

                elif 'Frame Time:' in line:
                    break
        self.end_sites = end_sites

        joint_parents = [-1] + [joints.index(i) for i in joint_parents[1:]]
        channels = [len(i) for i in channels]
        # print(end_sites)
        return joints, joint_parents, channels, joint_offsets, original_meta_data

    def load_motion_data(self, bvh_path):
        with open(bvh_path, 'r') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                if lines[i].startswith('Frame Time'):
                    break
            motion_data = []
            for line in lines[i+1:]:
                data = [float(x) for x in line.split()]
                if len(data) == 0:
                    break
                motion_data.append(np.array(data).reshape(1, -1))
            motion_data = np.concatenate(motion_data, axis=0)
        # print('motion_data.shape:', motion_data.shape)
        return motion_data

    def load_motion(self, bvh_file_path):
        '''
            读取bvh文件，初始化元数据和局部数据
        '''
        self.joint_name, self.joint_parent, self.joint_channel, joint_offset, self.original_meta_data = \
            self.load_meta_data(bvh_file_path)

        motion_data = self.load_motion_data(bvh_file_path)

        # 把motion_data里的数据分配到joint_position和joint_rotation里
        self.joint_position = np.zeros(
            (motion_data.shape[0], len(self.joint_name), 3))
        self.joint_rotation = np.zeros(
            (motion_data.shape[0], len(self.joint_name), 4))
        self.joint_rotation[:, :, 3] = 1.0  # 四元数的w分量默认为1

        cur_channel = 0
        for i in range(len(self.joint_name)):
            if self.joint_channel[i] == 0:
                self.joint_position[:, i, :] = joint_offset[i].reshape(1, 3)
                continue
            elif self.joint_channel[i] == 3 and '_end' not in self.joint_name[i]:
                self.joint_position[:, i, :] = joint_offset[i].reshape(1, 3)
                rotation = motion_data[:, cur_channel:cur_channel+3]
            elif self.joint_channel[i] == 3 and '_end' in self.joint_name[i]:
                self.joint_position[:, i, :] = joint_offset[i].reshape(1, 3)
                rotation = [0, 0, 0]
                cur_channel -= 3
            elif self.joint_channel[i] == 6:
                self.joint_position[:, i, :] = motion_data[:,
                                                           cur_channel:cur_channel+3]
                rotation = motion_data[:, cur_channel+3:cur_channel+6]

            self.joint_rotation[:, i, :] = R.from_euler(
                self.euler, rotation, degrees=True).as_quat()
            cur_channel += self.joint_channel[i]

        return

    def batch_forward_kinematics(self, joint_position=None, joint_rotation=None):
        '''
        @brief: 批量计算全局坐标系下的关节位置和旋转，存储到 self.joint_translation 和 self.joint_orientation
        @param joint_position: (N,M,3)的ndarray, 局部平移
        @param joint_rotation: (N,M,4)的ndarray, 用四元数表示的局部旋转
        @return joint_translation: (N,M,3)的ndarray, 全局平移
        @return joint_orientation: (N,M,4)的ndarray, 用四元数表示的全局旋转
        '''
        if joint_position is None:
            joint_position = self.joint_position
        if joint_rotation is None:
            joint_rotation = self.joint_rotation

        joint_translation = np.zeros_like(joint_position)
        joint_orientation = np.zeros_like(joint_rotation)
        joint_orientation[:, :, 3] = 1.0  # 四元数的w分量默认为1

        # 一个小hack是root joint的parent是-1, 对应最后一个关节
        # 计算根节点时最后一个关节还未被计算，刚好是0偏移和单位朝向

        for i in range(len(self.joint_name)):
            pi = self.joint_parent[i]
            parent_orientation = R.from_quat(joint_orientation[:, pi, :])
            joint_translation[:, i, :] = joint_translation[:, pi, :] + \
                parent_orientation.apply(joint_position[:, i, :])
            joint_orientation[:, i, :] = (
                parent_orientation * R.from_quat(joint_rotation[:, i, :])).as_quat()
        self.joint_translation = joint_translation
        self.joint_orientation = joint_orientation
        return joint_translation, joint_orientation

    def get_T_pose(self):
        translation = np.zeros((len(self.joint_name), 3))
        for i in range(len(self.joint_name)):
            pi = self.joint_parent[i]
            translation[i, :] = translation[pi, :] + \
                self.joint_position[0, i, :]
        return translation

    def adjust_joint_name(self, target_joint_name):
        '''
        调整关节顺序为target_joint_name
        '''
        idx = [self.joint_name.index(joint_name)
               for joint_name in target_joint_name]
        idx_inv = [target_joint_name.index(joint_name)
                   for joint_name in self.joint_name]
        self.joint_name = [self.joint_name[i] for i in idx]
        self.joint_parent = [idx_inv[self.joint_parent[i]] for i in idx]
        self.joint_parent[0] = -1
        self.joint_channel = [self.joint_channel[i] for i in idx]
        self.joint_position = self.joint_position[:, idx, :]
        self.joint_rotation = self.joint_rotation[:, idx, :]
        pass

    def raw_copy(self):
        '''
        返回一个拷贝
        '''
        return copy.deepcopy(self)

    def sub_sequence(self, start, end, step=1):
        '''
        返回一个子序列
        start: 开始帧
        end: 结束帧
        '''
        res = self.raw_copy()
        res.joint_position = res.joint_position[start:end:step, :, :]
        res.joint_rotation = res.joint_rotation[start:end:step, :, :]
        return res

    def append(self, other):
        '''
        在末尾添加另一个动作
        '''
        other = other.raw_copy()
        other.adjust_joint_name(self.joint_name)
        self.joint_position = np.concatenate(
            (self.joint_position, other.joint_position), axis=0)
        self.joint_rotation = np.concatenate(
            (self.joint_rotation, other.joint_rotation), axis=0)
        pass

    def print_dims(self):
        print('joint num:', len(self.joint_name))
        print('end joint:', self.end_sites)
        print('motion length:', self.motion_length)
        print(f'joint_position.shape:{self.joint_position.shape}')
        print(f'joint_rotation.shape:{self.joint_rotation.shape}, note: you may find the end_sites still have rotations, but actually their rotations are 0 and won\'t be outputed if you call \'put_back_bvh()\', it\'s just to occupy a position')
        print("")
        pass

    def put_back_bvh(self, suffix='put_back'):
        '''
        @brief: 把joint_position和joint_rotation写回bvh文件
        @note: 数据来源是 self.joint_position 和 self.joint_rotation ，都是局部坐标系
        @param suffix: 文件名后缀
        @return: None
        '''
        put_back_file_path = '.'.join(self.bvh_file_path.split(
            '.')[:-1]) + '-' + suffix + '.bvh'
        with open(put_back_file_path, 'w') as f:
            for line in self.original_meta_data:
                if 'Frames' in line:
                    f.write('Frames: %d\n' % self.motion_length)
                else:
                    f.write(line)

            for i in range(self.motion_length):
                f.write(
                    ' '.join(['%6f' % x for x in self.joint_position[i, 0, :].flatten()]))
                f.write(' ')
                joint_rotation_without_end_site = np.array([elem for idx, elem in enumerate(
                    self.joint_rotation[i]) if idx not in self.end_sites])
                f.write(
                    # ' '.join([' '.join(['%6f' % y for y in R.from_quat(x).as_euler(self.euler, degrees=True)]) for x in self.joint_rotation[i, :, :]]))
                    ' '.join([' '.join(['%6f' % y for y in R.from_quat(x).as_euler(self.euler, degrees=True)]) for x in joint_rotation_without_end_site]))
                f.write('\n')
        pass

    def get_mid_pos(self, fluctuation_rate=0.):
        '''
        @brief: 计算全局坐标系下的骨骼中点位置
        @note: 前提是已经计算了全局坐标系下的关节位置和旋转
        @param fluctuation_rate: 随机扰动的比例
        @return joint_markers: (N,M,3)的ndarray, 全局骨骼中点
        '''
        joint_markers = np.zeros_like(self.joint_position)
        self.marker_fluctuation_rate = np.random.uniform(-fluctuation_rate, fluctuation_rate, (self.joint_num))

        for i in range(len(self.joint_name)):
            if i == 0:
                continue
            pi = self.joint_parent[i]
            parent_orientation = R.from_quat(self.joint_orientation[:, pi, :])
            joint_markers[:, i, :] = self.joint_translation[:, pi, :] + \
                parent_orientation.apply(self.joint_position[:, i, :])/2 * \
                (1+self.marker_fluctuation_rate[i])

        print("marker_fluctuation_rate:\n", self.marker_fluctuation_rate)
        print("")
        return joint_markers

    @property
    def motion_length(self):
        return self.joint_position.shape[0]

    @property
    def joint_num(self):
        return len(self.joint_name)
