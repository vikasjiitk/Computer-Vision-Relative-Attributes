Feature Extraction
==========
The code is taken from: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

You can change the sourcedir to get the features at the required place. It takes around 1 second for one image.

###Requirements:

Theano - Latest (our version - 0.8.0)
Keras - Latest (our version - 0.3.3)

###About:
This is the Keras model of the 16-layer network used by the VGG team in the ILSVRC-2014 competition.
It has been obtained by directly converting the Caffe model provived by the authors.
Details about the network architecture can be found in the following arXiv paper:

Very Deep Convolutional Networks for Large-Scale Image Recognition
K. Simonyan, A. Zisserman
arXiv:1409.1556
In the paper, the VGG-16 model is denoted as configuration D. It achieves 7.5% top-5 error on ILSVRC-2012-val, 7.4% top-5 error on ILSVRC-2012-test.

###Execution
Download weight file of CNN model from: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
Create  a folder containing images of which you want to extract features of. Change the path of sourceDir in the vgg-16_keras.py.  
Run the command:
```
python vgg-16_keras.py
```
Features will be saved in features.csv file
