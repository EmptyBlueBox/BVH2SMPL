# BVH2SMPL

## Update on May 24th, 2024

After step 2, you can try to use [aitviewer](https://github.com/eth-ait/aitviewer) to render the video rather than using a lot of `.obj` files.

There's a reference script on [Inter-X repo](https://github.com/liangxuy/Inter-X/blob/main/visualize/smplx_viewer_tool/data_viewer.py)

## Usage

### Install the dependencies:

```shell
conda create -n animation python=3.10.12
pip install -r requirements.txt
```

### Download model:

Goto [SMPL](https://smpl.is.tue.mpg.de) 

Download this one: `[Download version 1.0.0 for Python 2.7 (female/male. 10 shape PCs)`

Rename the neutral model into `SMPL_NEUTRAL.pkl` and put it into `./src/rendering_utils/smpl`

### Goto soruce code folder:

```shell
cd ./src
```



### Run the script:

1. Put the bvh files into `./bvh` folder, then run script to preprocessing the files.

   To get the trajectory ends at the origin, I calculate the Y rotation of the root point at the last frame, then cast the inverse Y-rotation to the root rotation and root translation of all frames.
   
   You'll have to find out the time window the people is about to sit (put the bvh into blender and find a proper frame), and put the frame upper and lower range into the script.
   
   ```shell
   python preprocess.py
   ```
   
   

2. Run the script to retarget the motion

    ```shell
    python retargeting.py
    ```

    

3. Run the script to render the result

   note: in this part you must have cuda.
   
   ```shell
   python rendering.py
   ```

4. run the script to generate the video

   Goto `./results-objs` folder, then create a blender file, click on the bottom left, choose  `Text Editor` , then click `new` to create a new script, paste `./results-objs/script-objs_to_animation-in_blender.py` and run it. Like the picture.

   <img src="./README.assets/CleanShot 2024-03-10 at 06.32.14@2x.png" alt="CleanShot 2024-03-10 at 06.32.14@2x" style="zoom:30%;" />

   Then click the `Output` , choose 60fps (derived from the bvh), choose the output folder, choose the File Format as `FFmpeg Video` . 

   <img src="./README.assets/CleanShot 2024-03-10 at 06.31.45@2x.png" alt="CleanShot 2024-03-10 at 06.31.45@2x" style="zoom:30%;" />

   Then click `Render → Render Animation` , wait for some time and you will get your video in the output folder.

   <img src="./README.assets/CleanShot 2024-03-10 at 06.33.57@2x.png" alt="CleanShot 2024-03-10 at 06.33.57@2x" style="zoom:30%;" />

## Goal

I propose an end-to-end method to convert bvh file to SMPL mesh representation. 

And you can also simply modify the `retargeting.py` file to fit another bvh hierarchy.

## Results

<video width="320" height="240" controls>
  <source src="result-animation/All.mp4" type="video/mp4">
</video>

Goto `./results-animation` folder to see more results.

## Quotation

Code of step 3 is mostly from [motion-diffusion-model](https://github.com/GuyTevet/motion-diffusion-model)

## Epilogue

If you want to know how I complete this work, you may go to `Epilogue.md` to check it out.
