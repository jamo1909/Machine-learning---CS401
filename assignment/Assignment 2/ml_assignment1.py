# Instruction:
# Setting up Python enviroment 
# 1. Download Anaconda Python package for your platform(https://www.anaconda.com/)
# 2. Install Anaconda graphical GUI
# 3. open terminal
#	- install theano 
#		command:(conda install theano)
#	-Install the TensorFlow deep learning library 
#		command:(conda install -c conda-forge tensorflow)
#	-Install Keras 
#		command:(pip install keras)
# 4. In terminal call, run python program
#	command: (python ml_assignment1.py)

#FILES:
# train.txt - train python on this file
# test.txt - test python on this file
# key.txt - expected output for test.txt
# labels.txt - my output 
# terminal.txt - terminal code

# Machine Learning Assignment 1 - James Murphy(16421512)
from numpy import genfromtxt
from keras.models import Sequential
from keras.layers import Dense

# load the training set from train.txt
trainSet = genfromtxt(r"train.txt",delimiter=" ")
# load the test set from text.txt
testSet = genfromtxt(r"test.txt",delimiter=" ")
X = trainSet[:,0:10] #Collect the first 10 items from train.txt
y = trainSet[:,10] #Collect expected output at the 11th position
Z = testSet[:,0:10] #Collect test set from first 10 positions
#keras model
model = Sequential()
model.add(Dense(64, input_dim=10, activation='tanh'))
model.add(Dense(32, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
# compile keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the trainingset
model.fit(X, y, epochs=250, batch_size=10)
# evaluate 
_, accuracy = model.evaluate(X, y)
#Print accuracy
print('Accuracy: %.2f' % (accuracy*100))
# make class predictions with the model
predictions = model.predict_classes(Z)
file = open("labels.txt","w")
# test the 10000 cases
for i in range(10000):
	print('%s => %d' % (Z[i].tolist(), predictions[i]))
	file.write('%d\n' % (predictions[i]))
file.close() 
# numberCorrectAnswers()


# def numberCorrectAnswers():
	#Comparing labels.txt to key.txt (Waiting on key.txt)
# 	print('Comparing labels.txt to key.txt')
# 	f1=open("labels.txt","r")
# 	f2=open("key.txt","r")
# 	total=0
# 	correct=0
# 	incorrect=0
# 	for line1 in f1:
# 	    for line2 in f2:
# 	        if line1==line2:
# 	            correct += 1
# 	        else:
# 	            incorrect += 1
# 	        break
# 	f1.close()
# 	f2.close()
# 	print('Percentage correct: %.2f' % ((correct/10000)*100))

