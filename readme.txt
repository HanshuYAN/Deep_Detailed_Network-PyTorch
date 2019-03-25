This is a re-implementation of our CVPR paper [1] and for non-commercial use only. You need to install Python with Tensorflow-GPU to run this code.

Install Tensorflow£º https://www.tensorflow.org/install/


Usage:


1. Preparing training data: put rainy images into "/TrainData/input" and label images into "/TrainData/label". Note that the pair images' indexes **must be** the same.

2. Run 
"training.py" for training and trained models should be generated at "/model".

3. After training, run 
"testing.py" to test new images.


We release our rainy image dataset at:  https://xueyangfu.github.io/projects/cvpr2017.html


If this code and dataset help your research, please cite our related papers:

[1] X. Fu, J. Huang, D. Zeng, Y. Huang, X. Ding and J. Paisley. ¡°Removing Rain from Single Images via a Deep Detail Network¡±, CVPR, 2017.

[2] X. Fu, J. Huang, X. Ding, Y. Liao and J. Paisley. ¡°Clearing the Skies: A deep network architecture for single-image rain removal¡±, IEEE Transactions on Image Processing, 2017.


Welcome to our homepage:   https://xmu-smartdsp.github.io/



