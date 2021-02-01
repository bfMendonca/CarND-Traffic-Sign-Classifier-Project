# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/histogram.jpg "Visualization"
[image2]: ./output_images/Testing_sample.jpg "Training images sample"
[image3]: ./output_images/Training_sample.jpg "Validation images sample"
[image4]: ./output_images/Validation_sample.jpg "Testing images sample"

[image5]: ./new_test_images/1.jpg "Testing images sample"
[image6]: ./new_test_images/3.jpg "Testing images sample"
[image7]: ./new_test_images/5.jpg "Testing images sample"
[image8]: ./new_test_images/6.jpg "Testing images sample"
[image9]: ./new_test_images/7.jpg "Testing images sample"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data count for each label

![alt text][image1]

And here is a sample of the images contained at the Training, Validation and Test sets

![alt text][image2]

![alt text][image3]

![alt text][image4]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a step to increase the neural network performance, also described in detail in this blog, https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/, it seems that is best to normalize the input so that each pixel is of type float and contained in the range of 0.0 and 1.0.

Taking as input an RGB Image, with three color channels, and that each channel is within the 0, 255 range, each color channel for each channel was divided with the value of 255 in order to constrain it's value in the range described above. 

The described proccess is at the IPython notebook "Traffic_Sign_Classifier.ipynb" at the 4th code block cell. 

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The final model considered was the standart LeNET architecure, but altered so taht it has 3 color channel as inputs

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16					|
| Convolution 5x5	    | 1x1 stride,  valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling			| 2x2 stride,  outputs 5x5x16					|
| Flatten				| outputs 400									|
| Fully connected		| inputs 400,  outputs, 120						| 
| Dropout 				| 												| 
| RELU					|												|
| Fully connected		| inputs 120,  outputs, 54						| 
| Dropout 				| 												| 
| RELU					|												|
| Output 				| 43 classes									|
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The model training used the Adam optmizer with the follow configurations:

EPOCHS = 100
BATCH_SIZE = 64
rate = 1e-3
drp = 0.6

Here, drp stands for the droupout.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.00
* validation set accuracy of 0.972
* test set accuracy of 0.953

The first architecture choosen was based on the LeNET-5 architecture, with minor additions regarding the droupout added between the fully connected layers. As the LeNET was orignally conceived fir digit recognition, it semed to be a good candidate. The Accuracys reported above seemed to be a good indicative that the network is efficient in Traffic sign classification.  

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| No-vehicle 			| No-vehicle									|
| Speed limit 30km/h	| Speed limit 30km/h							|
| Speed limit 50km/h	| Speed limit 50km/h			 				|
| Children Crossing 	| Children Crossing    							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 10th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| Bycicle Crossing								|
| .05					| Speed limit 30km/h							|
| .04	      			| Yield 						 				|
| .01				    | No entry  									|

For the second image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| No vehicle   									| 
| .20     				| Speed limit 50km/h							|
| .05					| Yield											|
| .04	      			| Keep right					 				|
| .01				    | Priority road									|

For the third image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Speed limit 30km/h							| 
| .20     				| Priority road									|
| .05					| Speed limit 50km/h							|
| .04	      			| End of speed limit 80km/h		 				|
| .01				    | Rondabout mandatory  							|

For the fourth image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Speed limit 50km/h							| 
| .20     				| Speed limit 30km/h							|
| .05					| Speed limit 80km/h							|
| .04	      			| Double curbe					 				|
| .01				    | Speed limit 60km/h   							|

For the fifth image ... 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Children Crossing								| 
| .20     				| Bycicles Crossing								|
| .05					| Beware Ice/Snow								|
| .04	      			| Bumpy Road					 				|
| .01				    | Dangerous cruve to the right					|


