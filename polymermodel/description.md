## Model versions description

`version001`

Fits an image of a curve (convoluted with some gaussian sigma) , assuming we know the length.
Parameters are curve's number of segments, segment length and the anchor point position.

A simple curve with a relatively small bending case is in version001.py` and `MAX_GT_all-1.tif` as an input.

A highly bent curve is in version001_snake.py` and `snake.tif` as an input. 
In this case it fails. But maybe using the trick of different sigma, i.e. first going 
from large values to small (appears later) can help here.

`version002`

Simulates many "rolling shutter" acquisition datasets and tries to find model parameters.
Now for 2D only.
Fails at the highly oscillating end of the curve.

`version003`

Tries to simulate "z-slice" async acquisition as ground truth.
But instead of polymer chain model does some weird simplification, not useful.

`version004`

Better simulation of async per slice (row) data, used as input.
But the solution converges to half of the original frequency.

`version005`

Uses a slightly higher learning rate for the frequency (omega) to escape local harmonic traps.
Recovers the ground truth, but I think at this moment does not use async acquisition data,
just a beating with some single delay and "global shutter".

`version006`

Uses GT generator generated single "rolling shutter" and a set of "async" per slice/row acquisitions
as an input and recovers the position.

`version007`

Loads provided ground truth images and looks for solution.
Now as an input a single "rolling shutter" image is provided (`2D_rolling_shutter.tif`)
and a stack of async slices/rows (`2D_async.tif`). 
Uses convolution sigma optimization (large at initial steps, smaller later).
Can change the K modes in angle description.
Can setup initial frequency. The cilia root is fixed and provided.
Can set hard and soft constrains on angles.

`version009`

First 3D version, loads the async data only (ground truth, `3D_data_async.tif`).
It is possible to change the voxel size in XY and Z (+ convolution sigma value proportionally).
Relative angle constrain + convolution sigma are changing during optimization.
Converges to ground truth "okeyish", but data far away from the root are still not ok.



