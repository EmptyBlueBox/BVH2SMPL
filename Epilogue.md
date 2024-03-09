# Epilogue



## Day 1

After trying for one night, fairmotion still cannot be installed because it seems that fairmotion requires torch==1.4.0 but I am unable to install such an old version. Therefore, I plan to use my own bvh_loader, which consists of only one Python script and suits my usage habits without many dependencies.

经过一晚上的尝试， fairmotion 仍然无法安装，因为似乎 fairmotion 需要 torch==1.4.0 但是我无法安装如此古老的版本，因此我打算使用我自己写的 bvh_loader，它只由一个python 脚本组成，符合我的使用习惯，也没有很多依赖项。

Next, I looked at `render_mesh.py`, and then was very puzzled as to why it used `sample##_rep##.mp4` as input, because `.mp4` files do not contain three-dimensional motion information. However, after reading the code in detail, I found that it only used the information in the filename of `sample##_rep##.mp4`, without reading the file content. The actual useful information is from `results.npy` which is located in the same path as this `mp4` file. But this `results.npy` was not explicitly used as an input, leading to a great deal of misunderstanding.

接下来我看了 `render_mesh.py` ，然后非常疑惑为什么它使用了 `sample##_rep##.mp4` 作为输入，因为 `.mp4` 文件不包含三维运动信息。但是详细阅读代码之后发现它仅仅使用了 `sample##_rep##.mp4` 的文件名中的信息，并没有读取文件内容。实际有用的信息是和这个 `mp4` 文件同路径下的 `results.npy` 。但是这个 `results.npy` 并没有显式地作为输入，因此导致了很大的误解。

Next, I attempt to analyze the structure of `results.npy` in order to use my own bvh_loader to generate the corresponding skeletal rotations. This requires me to analyze `/sample/generate.py`, where I discovered the following code:

接下来我尝试分析 `results.npy` 的结构以使用自己的 bvh_loader 生成对应的骨骼旋转。这要求我分析 `/sample/generate.py` ，发现了如下代码：

```python
np.save(npy_path,
        {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
        'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
```

This indicates that the useful data in `results.npy` is `result[motion]`, and there is also the following code in `visualize/vis_utils.py`:

这说明 `results.npy` 中有用的数据为 `result[motion]` ，同时在 `visualize/vis_utils.py` 还有如下代码：

```python
motion_tensor, opt_dict = self.j2s.joint2smpl(self.motions['motion'][self.absl_idx].transpose(2, 0, 1))  # [nframes, njoints, 3]
```

This indicates that `result[motion]` has four dimensions. The first dimension is the model-related `batch_size`, which we do not need; we only require the last three dimensions. The first dimension is the number of frames, the second dimension is the number of joints, and the third dimension is the rotation angle of each joint. However, what still needs to be addressed is the order of joints.

这说明 `result[motion]` 共有四个维度，第一个维度是模型相关的 `batch_size` ，我们并不需要这个维度，只需要它的后三维即可，第一维是帧数，第二维是关节个数，第三维是关节的旋转角。但是仍需要处理的是关节顺序。

Then I directly analyzed the `.bvh` file of `SMPL` to understand what should be done during retargeting. I went to [smpl](https://smpl.is.tue.mpg.de/index.html) and downloaded the `.fbx` sample file, then exported it as a `.bvh` file. After comparing and analyzing, I found that the bone structures of src and dst are not exactly the same. `SMPL` has one less `chest` joint but more `hand` joints, one for each hand. Therefore, in the process of retargeting, I need to remove any one of the `chest` joints because it seems that they have little influence on these actions; add two `hand` joints which can simply copy their parent node's rotation since it appears to have little impact.

然后我直接分析了 `SMPL` 的 `.bvh` 文件，以弄清 retargeting 的时候应该怎么做。我前往 [smpl](https://smpl.is.tue.mpg.de/index.html) 下载了 `.fbx` 的示例文件，然后将其导出为 `.bvh` 文件，对比分析之后我发现 src 和 dst 的骨骼结构不完全一样。`SMPL` 比给出的骨骼结构少了一个 `chest` 关节，多了 `hand` 关节，两只手各一个。因此在 retargeting 的过程中，我需要删除任意一个 `chest` 关节，因为看起来在这些动作中 `chest` 关节影响不大；添加两个 `hand` 关节，可以直接复制其父节点的旋转，因为看起来影响不大。



## Day 2

I continued to analyze `visualize/vis_utils.py` to obtain the joint order, but after recursively searching all relevant files, it only indicated that there are 24 joints and three coordinates representing the root's offset. However, after reading information related to smpl, I found that smpl indeed has the same number of nodes.

我继续分析了 `visualize/vis_utils.py` 以获取关节顺序，但是在递归地查找了所有相关文件后只说明了关节个数为24，还有三个坐标代表root的offset这个信息，但是我在阅读 smpl相关信息后发现smpl确实和这个节点个数相同。

Then I temporarily gave up on solving this problem and turned to complete the code for preprocessing and retargeting.

然后我暂时放弃了解决这个问题，转而完成 preprocessing 和 retargeting 的代码。

Because I gave up using fairmotion and instead used the bvh_loader that I optimized according to my own usage habits in previous projects, these two parts of the code were completed in less than an hour. The processed three BVH files with a length of 300 are saved in `./bvh/*-preprocessing.bvh`.

因为我放弃使用 fairmotion，转而使用我在以往项目中自己按照自己的使用习惯优化的 bvh_loader ，所以这两部分的代码不到一个小时就完成了，处理完的三个长度为 300 的 bvh 文件保存在了 `./bvh/*-preprocessing.bvh` 中。

Next, all that's left is how to render this mesh.

接下来只剩下如何渲染这个mesh了。

The former [motion diffusion model](https://github.com/GuyTevet/motion-diffusion-model) might not good to use, instead I found [joints2smpl](https://github.com/wangsen1312/joints2smpl/tree/main) might be better, but it needs numpy<=1.22; furthermore I found [body-model-visualizer](https://github.com/mkocabas/body-model-visualizer/tree/master) and [smpl renderer](https://github.com/YunYang1994/OpenWork/tree/master/smpl) might be useful.

原始的 [motion diffusion model](https://github.com/GuyTevet/motion-diffusion-model) 可能不好用，我发现了 [joints2smpl](https://github.com/wangsen1312/joints2smpl/tree/main) 或许可以使用，但是需要将 numpy 降级至 1.22.x；另外我发现 [body-model-visualizer](https://github.com/mkocabas/body-model-visualizer/tree/master) 和 [smpl renderer](https://github.com/YunYang1994/OpenWork/tree/master/smpl) 或许也可以用。

After trying all the above projects, I successfully rendered an image using [smpl renderer](https://github.com/YunYang1994/OpenWork/tree/master/smpl), and it looks pretty good with lighting added, but it can only render a single frame.

在尝试过以上所有项目之后我使用 [smpl renderer](https://github.com/YunYang1994/OpenWork/tree/master/smpl) 成功渲染出了图片，加入了光照之后看起来效果不错，但是只能渲染一帧。

Since the above method can only generate 2D images, I eventually went back to [motion diffusion model](https://github.com/GuyTevet/motion-diffusion-model), and then I recursively found all the code related to visualization. After at least three hours of arduous testing and debugging, I finally got it running.

由于上述方法只能生成2D图片，我最终还是回到了 [motion diffusion model](https://github.com/GuyTevet/motion-diffusion-model) ，然后我递归地找出了所有有关 visualize 的代码，经过至少三个小时的艰苦卓绝的测试和debug，终于跑通了。

Optimization time: Three action clips, each with 300 frames, took about 25 minutes to optimize on a `4090` graphics card.

优化时间：三个动作片段各300帧，优化时间在 `4090` 显卡上耗时约25分钟。

However, I found that the end nodes, such as the head and limbs, had a large margin of error. Then I realized in `customloss.py`, in fact, when optimizing the mesh only four nodes were being optimized, which resulted in poor performance. Therefore, I modified `customloss.py` to include optimization for the end nodes. In total 11 joints were optimized out of 24 joints across the whole body.

但是我发现末端节点，比如脑袋和手脚误差很大，然后我发现在 `customloss.py` 中，其实在优化 mesh 的时候只优化了四个节点的 mesh，所以效果很差，因此我修改了 `customloss.py` ，加入了对末端节点的优化，总计11个关节，全身24个关节。

I've found that the unit for smpl is meters, while the unit for src bvh is centimeters. Even after fixing this bug, the results are still very poor.

我又发现 smpl 的单位是米，src bvh单位是cm，修改了这个bug之后效果仍然很差。

Then I realized that what smplify optimizes is actually joint rotation, so why don't I just use the existing joint rotations? It saves time on optimization and yields better results.

然后我发现 smplify 优化的其实就是关节旋转，那我为什么不直接用现成的关节旋转呢，既不用浪费时间优化，效果又好。

SUCCESS!

成功了！