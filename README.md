
# Traffic Sign Recognition
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


## Project 3 (Self-Driving Car Engineer - Udacity)




**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./exploratory.png "Visualization"
[image2]: ./initial_histo.png "Initial Histogram"
[image3]: ./augment.png "Augment"
[image4]: ./final_histo.png "Final Histogram"
[image5]: ./Architecture.png "Architecture"
[image6]: ./prediction.png "Traffic Signs"


Here is a link to my [project code](./Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Summary of the data set.

The files provided contained training and validation datasets, initially: 

* The size of training set is 34799
* The size of the validation set is 12630
* The shape of a traffic sign image (RGB) is (32, 32, 3) 
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. On top of each image you can see the label corresponding to the traffic sign, you can find the meaning of each one by looking at the file [signnames.csv][./signnames.csv]

![alt text][image1]

I plotted the initial distribution of the labels to verify how balanced they are. 

![alt text][image2]


### Design and Test a Model Architecture

#### 1.  Image data preprocessing. 

As a first step, I decided to balance the classes by applying data augmentation techniques as, flipping, rotation, saturation modification and light manipulation. The main idea behind this process is to have nearly 1500 images per label. Labels with less than 1500 images were supposed to be augmented to reach that number 25% per each technique. For practical usage and computational constraints, I decided only to augment till minimum 100 images per label (but this can be changed easily in the code).
Here is an example of a traffic sign image with different augmentation techniques.

![alt text][image3]

After applying the data aumentation, the histogram of the training set looks like this: 

![alt text][image4]

This step takes quite a lot of time since we're trying to have an equal number of samples per label.

I tested a CNN architecture using RGB images and I realised I needed a larger network and hence the training time would be too much for my computer. So, I decided to convert the images to grayscale.

As the last step, I normalized the image data to change the values of the images in the dataset to use a common scale, and have a distribution similar to a gaussian without distorting differences in the ranges of values or losing information.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

![alt text][image5]

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| C1 Convolution 5x5     | 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|		|
| Max pooling	| 2x2 stride,  outputs 14x14x8 				|
|C3 Convolution 5x5  | 1x1 stride, valid padding, outputs 10x10x16  | 
| RELU |  |
|C4 Convolution 7x7	| 1x1 stride, valid padding, outputs 22x22x10| |
RELU||
|Max pooling | 2x2 stride,  outputs 11x11x10|
| C5  Convolution 2x2   | 1x1 stride, valid padding, outputs 10x10x16   |
|RELU | |
|Concatenation  | C5 and C3, outputs 10x10x32 |
| Max pooling| 2x2 stride,  outputs 5x5x32 	|
|Flatten| outputs 800|
|Fully connected | outputs 120|
|RELU|
|Fully connected| outputs 84 |
| Fully connected | outputs 43        									|
| Softmax				|         								
 


#### 3. Model training. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an learning rate of 0.001 with and Adam Optimizer, based on literature Adam learns the fastes, while it is more stable than the other optimizers (it doesn’t suffer any major decreases in accuracy).
 I set the epochs to 50 
 Batch size was set to 100 samples
 I set the dropout to 50%


#### 4. Approach

I tried different architectures, it was a bit tricky since retraining takes a lot of time, but finally I got an accuracy of 0.94 in the validation set. 

My final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.94
* test set accuracy of 0.929

I tried multiple variations of LeNet that I made. The very first approach was using RBG images, the main problem with it was that the accuracy was never higher than 60%, I presume that for RGB images I'd need a deeper network, so I add some convolutional layers more, and tried the inception approach, but the accuracy didn't improve much. Due to time and computational contrains I didn't explore further, but I decided to use images in gray scale instead. With the initial approach the accuracy wasn't going above 90% even when the images were in grayscale, so I decided to start removing some layers. From the previous section we can see that C2 is missing, that corresponds to a 1x1 convolutional layer that was removed. After this change I was able to train the network achiving and accuracy above 93%.  
 

### Test a Model on New Images

I downloaded eight German traffic signs from the web and run the CNN to test it over the new data. 
Here you can find the results:
![alt text][image6] 


As we can see in the previous image, the predictions were correct for 7 of the 8 testing images (accuracy in the new set of images 0.875). The sixth image it's difficult to classify even for a human. The number 80 in the image can be easily be confused with a 60, which was the prediction of the network. By looking at the softmax probabilities of the CNN for the testing images we can infer that 6th image is the only image with two clear peaks, being the second one the correct prediction.  


