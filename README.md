# Image Classification using Nearest Neighbors
Solving simple image classification problem using KNN classifier algorithm.

## Introduction

Why use KNN when current CNN models can give a better accuracy?
If you were given a problem that is to classify an image from a set of labels like cat or dog, what would your initial approach be? and can you better this approach any more?
these questions you can ask when you know the developments from the first model. So start your computer vision with this.

If you are reading this then you must be new to CV, so we will be using CIFAR10 Dataset which contains 50,000 training images and 10,000 test images of 10 classes, each image 32 by 32 pixel and 3 color channels. Our aim here is to create a model which has accuracy more than 10%, why? Because random choice of a class in 10 classes is 10%.

Let's start to break down the problem and find a way to solve it.

## Image Classification

If you look at an image of a cat, you can process and categorize it as a cat, very quickly and human accuracy of image classification is about 94%.
Same image the computer views it as a multidimensional array of pixel values. Like in our dataset we have 50,000 training images each of 32*32 pixels and each pixel with 3 color values.
So the ndarray would probably have this shape (50,000, 32, 32, 3)

But even if it's a computer what challenges can it face?
Image can be oriented in different ways, tilted or rotated. Then it might be zoomed in or out. It can be deformed or maybe a part of the whole object. Or the illumination or background troubling the classification.

So many challenges but let's start with a simple 3 class classification using Numpy.

## Sample Problem

Open either jupyter notebook or vscode, any will do and make sure to install pandas and numpy before starting.

```python
import numpy as numpy
import pandas as pd 

# assume a training set
train = pd.DataFrame({'a':[1,10,300], 'b':[1,50,600], 'c':[0,80,900], 'target':['cat', 'dog', 'car']})
print(train)

# assume a test set with 2 images to classify
test = pd.DataFrame({'a':[15,2], 'b':[45,1], 'c':[50,4]}) # by looking at this, 1st image is dog and next cat

pred = []
# so everything is fine till now

train_data = train[['a','b','c']]
print(train_data)

for i in range(test.shape[0]):
    x = np.reshape(test.iloc[i], 3)
    
    ab_diff = np.abs(train_data - x)

    net = np.sum(ab_diff, axis=1)

    min_index = np.argmin(net)

    pred.append(train['target'][min_index])

print(pred)
# see output is dog and cat
```
Initially I created a sample training dataset where a,b,c are pixel values and I have 3 rows that means assume 3 images.
They are labeled as ['cat', 'dog', 'car'] respectively. I made it clear that each value range is different to show they are different classes.
Like [1,1,0] maps to 'cat' so we understand cat has a low values.

We create a test set with 2 images to classify. To start the algorithm we need to check if the values of test image is closer to any of the vlues on the training set.
So in this example we make total 2*3 comparisions. But in our cifar10 dataset we got to make 10,000*50,000 comparisons. Which takes a lot of computational time.

Training time is constant as all it has to do is memorize the training dataset.
Validation time varies on number of images in training dataset.

Anyway this is the most complicated part, computation using numpy arrays.
We create a prediction list to append the model predicts.
First we take the first image in test set using pandas integer loc function. We then reshape it flat, basically laying it in a row but reducing the dimension.
Numpy function np.abs(input), outputs an array of size of input so the absolute values of each differnce in pixels are calculated and stored in a 2d array, why?
because we are making 3 image comparisions here so 9 values, 3 by 3 matrix.

Next step is to sum the error terms for each computation so you have to add horizontally hence we use Numpy function np.sum()
and enter the axis parameter as 1, which specifies to add horizontally and iterate vertically.

So you now have the errors of each comparision, the lowest error means the test image belongs to that specific training image.
Next function outputs the index of min value in the array, hence we get the index of the training image and we can index the class similarly.

And there you go, you got the classes and it does seem correct right.


