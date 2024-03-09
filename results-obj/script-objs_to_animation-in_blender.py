import bpy
import os
import re

def sort_key(obj_file):
    """提取文件名中的数字用于排序"""
    numbers = re.findall(r'\d+', obj_file)
    if numbers:
        return int(numbers[0])
    return 0

fps = 60
obj_folder_path = bpy.path.abspath("//Armchair1/")
# obj_folder_path = bpy.path.abspath("//Desk1/")
# obj_folder_path = bpy.path.abspath("//Sit54/")
file_list = os.listdir(obj_folder_path)
# 按照文件名中的数字进行排序
file_list = sorted(file_list, key=sort_key)

# 初始化当前帧为0
frame_number = 0

# 遍历所有.obj文件
for obj_file in file_list:
    if obj_file.endswith(".obj"):
        # 导入.obj文件
        file_path = os.path.join(obj_folder_path, obj_file)
        bpy.ops.import_scene.obj(filepath=file_path)
        
        # 获取刚刚导入的对象
        imported_objects = bpy.context.selected_objects
        for obj in imported_objects:
            # 在当前帧之前设置对象为隐藏状态
            obj.hide_render = True
            obj.hide_viewport = True
            # 注意：这里我们需要在“frame_number - 1”设置隐藏关键帧
            # 这确保了在这一帧之前对象是不可见的
            obj.keyframe_insert(data_path="hide_render", frame=max(frame_number - 1, 0))
            obj.keyframe_insert(data_path="hide_viewport", frame=max(frame_number - 1, 0))
            
            # 在当前帧设置对象为可见
            obj.hide_render = False
            obj.hide_viewport = False
            obj.keyframe_insert(data_path="hide_render", frame=frame_number)
            obj.keyframe_insert(data_path="hide_viewport", frame=frame_number)
            
            # 在下一帧再次设置对象为隐藏状态
            obj.hide_render = True
            obj.hide_viewport = True
            obj.keyframe_insert(data_path="hide_render", frame=frame_number + 1)
            obj.keyframe_insert(data_path="hide_viewport", frame=frame_number + 1)
        
        # 进入下一帧
        frame_number += 1

# 设置动画播放的总帧数
bpy.context.scene.frame_end = frame_number
