# ML-Work
Machine Learning (ML) labs done of the course of my senior year in high school.

Within my "machine learning work folder," there are corresponding folders with the labels indicating which machine learning technique I used. 
Below is a description of what each file in each folder does. Other than the headers, I will refer to each folder to their index number.


# 0, 1, and 2: Preprocessing Learning
## "0 webdev" and "1 beginlabs"
The work within these folders are very fundamental. Within 0, there is simple html which is somewhat irrelevant to my machine learning work. Within 1, I practiced matplotlib plotting skills by plotting the data within the well-known file "iris.csv." This csv can be found on Kaggle with a Google search.

## "2 splittingdata"
The work within this folder are mainly located within the five files listed below. The other files are data I used to test splitting on (for test-train splits)
stratifytesttrain.py: Split the mushroom.csv file into 2 files-33% into one file for testing and 66% into another for training. This was done through stratified sampling rather than random sampling, so that the testing and training would have classes proportional to the original dataset.
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


