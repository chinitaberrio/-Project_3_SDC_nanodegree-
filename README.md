
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
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

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

![alt text][image1]


### Design and Test a Model Architecture

#### 1.  Image data preprocessing. 

As a first step, I decided to balance the classes by applying data augmentation techniques as, flipping, rotation, saturation modification and light manipulation. The main idea behind this process is to have nearly 1500 images per label. Labels with less than 1500 images were augmented to reach that number 25% per each technique.
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
|C2 Convolution 1x1 | 1x1 stride, valid padding, outputs 28x28x8 |
|RELU
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





#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


