# ML-Work
Machine Learning (ML) labs done of the course of my senior year in high school.

Within my "machine learning work folder," there are corresponding folders with the labels indicating which machine learning technique I used. 

### Note: The higher numbers correspond with generally higher level ML topics.

Below is a description of what each file in each folder does. Other than the headers, I will refer to each folder to their index number.


# 0, 1, and 2: Preprocessing Learning
## "0 webdev" and "1 beginlabs"
The work within these folders are very fundamental. Within 0, there is simple html which is somewhat irrelevant to my machine learning work. Within 1, I practiced matplotlib plotting skills by plotting the data within the well-known file "iris.csv." This csv can be found on Kaggle with a Google search.

## "2 splittingdata"
The work within this folder are mainly located within the five files listed below. The other files are data I used to test splitting on (for test-train splits)
- stratifytesttrain.py: Split the mushroom.csv file into 2 files-33% into one file for testing and 66% into another for training. This was done through stratified sampling rather than random sampling, so that the testing and training would have classes proportional to the original dataset.
- stratifytesttrain2.py: Similar to stratifytesttrain.py, but for the iris dataset.
- train-test-split.py: Used random sampling to split a dataset of 1000 points into test and train datasets. 33% was put into test and the rest was put into a training dataset.
- train-test-split2.py: Tested if the method from sklearn performed stratified sampling or not.
- train-test-split3.py: Testing for a repeatable split using sklearn.

## "2.1 Q1 Project Files"
Worked with a partner to create a mini-project where we took data from a public government source with data house representative votes on bills. Using that data, we classified whether or not a house representative was democratic or republican. Throughout the process, we had to preprocess the data, train a KNN, and write a paper. The code within this folder does not include the KNN because that was stored on my previous computer, however, I will add the paper report soon. In the end, we ended up with around 99% accuracy.

# 3: Different Machine Learning Algorithms
## "3.0 1R and Naive bayes"
Within this project there are 4 main files:
 - 1rlibraries.py: Coded a 1R network that would classify irises to their type using Python libraries
 - 1rscratch.py: Coded a 1R network from scratch without libraries to classify irises to their type
 - naivebayeslibraries.py: Coded a naive bayes network with libraries to classify irises
 - naivebayesscratch.py: Coded a naive bayes network from scratch without libraries to classify irises.

## "3.1 Decision Trees"
Within this project there are 2 main files:
- dt.py: Coded a Decision Trees network that would classify irises without using Python libraries
- dtLibraries.py: Coded Decision Trees network that would classify irises with using Python libraries

## "3.2 KNN"
Within this project there are 2 main files:
- knn.py: Coded a KNN that would classify irises without using Python libraries
- knnLibraries.py: Coded KNN that would classify irises with using Python libraries

## "3.3 K-Means"
within this project there are 3 main files:
- kmeans.py: Coded a K-Means network that would classify irises without using Python libraries
- kmeansLibraries.py: Coded K-Means that would classify irises with using Python libraries
- bruh.py: for some reason, my imports were returning errors so I was debugging within the file. The name "bruh" was an expression of my frustration.

# 4: More complicated machine learning algorithms

## "4.1 CNN" 
- Created a CNN that identified objects within an image (dogs vs cats). However, this work was performed on my better computer which I do not currently have access to, so I will have to upload it later. To see a better image classifier, please visit the "trash-boat" repository and view the Yolo-v5 network created to identify trash.

## "4.2 Q2 Project Stuff"
Another mini-project in which we modified KNN's weighting formula to see if we could gain greater results using more rounded curves in comparison to the typical straight line. Research paper will be posted at a later time when access is regained.

## "4.3 SVM" 
I created support vector machines: one of which to classify the iris dataset (irisSVM.py) and the other to classify random points a different classes (svm.py).

## "4.4 perceptrons"
Worked with creating perceptron networks in three different ways that correspond to the three files.
- pp1.py: trained perceptrons coded from scratch on 2 sets of points to classify each point as a point within either of the 2 separate clusters
- pp2.py: trained perceptrons coded from scratch on the iris dataset to classify each iris instance as its correct tyope
- pp3.py: Used Python libraries to create a multilayered perceptron network that was trained and tested on the iris dataset.

## "4.5 NN"
Created a NN using multilayered perceptrons to classify 10,000 points of data within the text file. There is only 1 important file, and that is backprop.py. The other files were tests I was doing in a separate file in order to "not mess up" the main backpropagation file. 
- backprop.py: Hard coded the backpropagation technique to create weights that would classify whether or not a given point was on a designated circle or not.

# 5: MNIST
Hardcoded a neural network that would take the pictures from the MNIST dataset (images of numbers; top result on Google search) in order to classify them as a number. I utilized backpropagation as well as the combination of forward pass, mean squared error, and backward pass in order to update the weights so that my neural network would be able to classify points with high accuracy. The MNIST code is heavily commented on the process, so feel free to take a look.


# 6 - 9 TBD


