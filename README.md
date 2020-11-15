# PFCNN: Convolutional Neural Networks on 3D Surfaces Using Parallel Frames
This repository contains the implementation of PFCNN introduced in our CVPR 2020 paper.

### Citation
If you find our work useful in your research, please consider citing:

    @inproceedings{yang2020pfcnn,
        title={PFCNN: Convolutional Neural Networks on 3D Surfaces Using Parallel Frames},
        author={Yang, Yuqi and Liu, Shilin and Pan, Hao and Liu, Yang and Tong, Xin},
        booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
        pages={13578--13587},
        year={2020}
    }
### Instalation

The code is divided in two parts: A C++ part for data preprocessing. The C++ part code is in pySurfaceHierarchy folder. Using pybind11, you can bind the C++ part with python part with a 'pyd' file. To compile the code you will also need to install following libraries:

- Openmesh : https://www.graphics.rwth-aachen.de/software/openmesh/
- Eigen : https://eigen.tuxfamily.org/dox/
- GeometricTools: https://www.geometrictools.com/
- ANN : https://www.cs.umd.edu/~mount/ANN/
- pybind11 : https://github.com/pybind/pybind11
- python : https://www.python.org/

After install these libraries, you need to specify the path to these libraries in pySurfaceHierarchy/CMakeLists.txt
Then compile the code with cmake, you will get a 'pySurfaceHierarchy.pyd' file in pySurfaceHierarchy/build/Release folder.

### Data Preprocessing
- Download the dataset for each task. Human_benchmark_sig_17.zip for human segmentation, MPI-FAUST_training.zip for human registration.

- Normalize the meshes, and config the data path in the preprocessing script.

- Copy the 'pySurfaceHierarchy.pyd' file to the 'PFCNN/- Generate_Dataset' folder. 'python37.dll', 'OpenMeshTools.dll' and 'OpenMeshCore.dll' is also needed. You can find them in the folders of download libraries above.

- Use pySurfaceHierarchy.hierarchy() and pySurfaceHierarchy.conv_para() to preprocess the dataset. You can follow the example in preprocess_data_faust_regression.py

- Then generate TFRecord dataset for network training. You can follow the example in FaustScanRegression_dataset.py

For the Faust Matching Task, run:

    python preprocess_data_faust_matching.py
    python FaustMatching_dataset.py
For the Faust Scan Regression Task, run:

    python preprocess_data_faust_regression.py
    python FaustScanRegression_dataset.py

The TFRecord we used for experiments will be uploaded soon.
### Train Network
Config the 'tfrecord_path' and file name in the specified args file.

You can follow the example in FaustScanRregression_args.py

For the Faust Matching Task, run:

    python FaustMatching.py --config FaustMatching_args
For the Faust Scan Regression Task, run:

    python FaustScanRregression.py --config FaustScanRregression_args
