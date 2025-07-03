# a_little_ML
playing around with some machine learning algorithms on a dataset that was made using googles icrawler, and mediapipe to process hand gestures.
Make sure to use PYTHON 3.10 because for now mediapipe isn't compatible well with newer versions.
# google's Icrawler
It is a multithreaded image/video capturer that can be configured, for our purpose we provided the max image size, the directory to put the images in, the 
category we would like to search for (thumbs up,thumbs down,hand gestures),we could even add some filters like, searching for real photos(exclusively) rather that 
having lots of emojis, also the images should be free to use meaning the licence should be noncomercial.

# Mediapipe and The Dataset
A Machine learning framework that provides pose detection, hand tracking , face meshes and much more. using this framework we can locate hands 
in the dataset and get 21 landmarks that correspond to each finger as well as the palms, we can use this data to create a data set that contains 
the euclidean distances from the WRIST to each landmark separately, so each hand with have 20 distances computed.

# Machine Learning Part
Just tried to find an excuse to use machine learning in a project, so i tried to use the generated dataset to train various machine learning 
models like artificial neural networks, K nearest neibours, support vectore machines, decision trees, and random forest, the most important part is 
changing the parameters of the models and checking the change in accuracy and time to finish.
This is just a simple hands on experiment that was intended to be for learning, it is not a serious attempt at making a robust mahine learning algorithm,
just some hands on experience.
