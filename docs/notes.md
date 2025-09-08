# Notes

- Due to legacy issues, the .obj and .stl files for the same object **differ by a 90 degrees rotation** around the x-axis.

- Direction convention: +x is right, +y is up, +z is backward. For the hand model with no rotation, the palm normal is (1, 0, 0) and the up normal is (0, 0, 1).

- The hand state is represented by a 25-dimensional vector: 3 for translation, 6 for rotation (see [this paper](https://arxiv.org/abs/2202.12555)), and 16 for joint angles.