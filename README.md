# Human Motion Retargeting

## Usage

Install the dependencies:

```shell
pip install -r requirements.txt
```

Run the script:

```shell
python main.py
```

## Goal

Retarget the motion represented by the bvh file onto the SMPL human mesh, and meanwhile modify these actions so that at their end frame, the root is at the origin, facing the same direction (in implementation, this direction is the positive x-axis).

## Method

## Results

## Acknowledgement

## Epilogue

Day 1

经过一晚上的尝试， fairmotion 仍然无法安装，因为似乎 fairmotion 需要 torch==1.4.0 但是我无法安装如此古老的版本。

接下来我看了 `render_mesh.py` 然后非常疑惑为什么它使用了 `sample##_rep##.mp4` 作为输入，因为 `.mp4` 文件不包含三维运动信息。但是详细阅读代码之后发现它仅仅使用了 `sample##_rep##.mp4` 的文件名中的信息，并没有读取文件内容。实际有用的信息是和这个 `mp4` 文件同路径下的 `results.npy` 。但是这个 `results.npy` 并没有显式地作为输入，因此导致了很大的误解，因此我认为这是 `motion-diffusion-model` 的源代码可读性太差，有待改进。

接下来我尝试分析 `results.npy` 的结构以使用自己的 bvh_loader 生成对应的骨骼旋转。这要求我分析 `/sample/generate.py` ，但是这个文件中的注释极少，可读性极差，所以我决定先放弃这个线索。

然后我直接分析了 `SMPL` 的 `.bvh` 文件，以弄清 retargeting 的时候应该怎么做。我前往 <https://smpl.is.tue.mpg.de/index.html> 下载了 `.fbx` 的示例文件，然后将其导出为 `.bvh` 文件，对比分析之后我发现 src 和 dst 的骨骼结构不完全一样。`SMPL` 比给出的骨骼结构少了一个 `chest` 关节，多了 `hand` 关节，两只手各一个。因此在 retargeting 的过程中，我需要删除任意一个 `chest` 关节，因为看起来在这些动作中 `chest` 关节影响不大；添加两个 `hand` 关节，可以直接复制其父节点的旋转，因为看起来影响不大。

之后我继续分析 `/sample/generate.py` ，发现了如下代码：

```python
np.save(npy_path,
        {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
        'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
```

这说明 `results.npy` 中有用的数据为 `result[motion]` ，同时在 `visualize/vis_utils.py` 还有如下代码：

```python
motion_tensor, opt_dict = self.j2s.joint2smpl(self.motions['motion'][self.absl_idx].transpose(2, 0, 1))  # [nframes, njoints, 3]
```

这说明 `result[motion]` 共有四个维度，第一个维度是模型相关的 `batch_size` ，我们并不需要这个维度，只需要它的后三维即可，第一维是帧数，第二维是关节个数，第三维是关节的旋转角。但是仍需要处理的是关节顺序。

Day 2

我继续分析了 `visualize/vis_utils.py` 以获取关节顺序，但是在递归地查找了所有相关文件后只得到了关节个数为22这个信息。

然后我暂时放弃了解决这个问题，转而完成 preprocessing 和 retargeting 的代码。
