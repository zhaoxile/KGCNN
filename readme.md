# KGCNN

This folder contains the implementation of the algorithm for image rain streaks removal named KGCNN.

## Usage:
If you want to generate dataset and train the network by youself, you need to generate training data first. The code is [there](https://github.com/zhaoxile/KGCNN/blob/master/data_generation/generate_data.py).
We provide the original images and code to generate the training data.
We also encourage you to generate their owen data for training or testing. 
Just replace the source image. 
And after understanding how the data is generated, you can construct their own data more conveniently.
After that, you could run [that](https://github.com/zhaoxile/KGCNN/blob/master/demo.py) for training and testing.

If you just want to use the trained model to test image, you need to run [that](https://github.com/zhaoxile/KGCNN/blob/master/demo.py) after modify function name from `train` to `load` on line 518 and 520.



## Reference

If you use this code, please cite

      @InProceedings{ ,
         author    = {},
         title     = {},
         booktitle = {},
         month     = {},
         pages     = {},
         doi       = {},
         year      = {}}

Contact: xlzhao122003@163.com\
Date: 19/07/2023