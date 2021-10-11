# active contour model -- level set methods

### Some popular level set methods with high citations are re-implemented with Python in this repository, including:
[1] [Active contours without edges (Chan_Vese model)](https://ieeexplore.ieee.org/abstract/document/902291) \
[2] [A Multiphase Level Set Framework for Image Segmentation Using the Mumford and Shah Model (multiphase Chan_Vese model)](https://link.springer.com/article/10.1023/A:1020874308076) \
[3] [Distance Regularized Level Set Evolution and its Application to Image Segmentation (DRLSE)](https://ieeexplore.ieee.org/document/5557813)
(This implementation was derived directly from this repo [link](https://github.com/Ramesh-X/Level-Set)) \
[4] [A Level Set Method for Image Segmentation in the Presence of Intensity Inhomogeneities With Application to MRI](https://ieeexplore.ieee.org/abstract/document/5754584) \
[5] [Active contours with selective local or global segmentation: A new formulation and level set method](https://www.sciencedirect.com/science/article/pii/S0262885609002303?casa_token=1XHVibcZy7YAAAAA:wbuTBb2m-0hpYq3nhDxau8ONLzwWwKcoi9wREMS7hiI555mRgGGdH4PdCZ07nk-hO2irL1w) \
[6-7] two toy implementations to combine the energy terms of **Chan_Vese model** and **DRLSE** to form the **region-based DRLSE** for two and multiphase segmentation.

### Requirements
The code was realized via Python 3.8, as well as some modules in the following.
```
numpy = 1.18.5
matplotlib = 3.2.2
scikit-image = 0.16.2
scipy = 1.5.0
```
### Execution
In the code, the level set initialization and parameters were set empirically, which may need to be tuned for different images.
Each method was wrapped into one folder, where demo.py could be executed to see the contour evolution.
```
python demo.py
```
