# Lab 9: Machine Learning

2.12/2.120 Intro to Robotics  
Spring 2024[^1]

In this lab, you will experiment with machine learning techniques on your own. Please submit a PDF of the screenshots and answers on Canvas. If you have any questions, feel free to reach out to the staff on Piazza.

## 1 Software Set Up

### 1.1 Scikit-Learn

To install Scikit-Learn, enter `pip3 install scikit-learn` in your terminal.

### 1.2 Tensorflow

To install Tensorflow, enter `pip3 install tensorflow` in your terminal.

## 2 Support Vector Machine (SVM)

As you may recall from lecture, SVMs is a suprevised learning method that can be used to separate data based on features. A support vector is a vector that essentially draws the boundaries between these classes based training using known, labeled data points.

### 2.1 Linearly Separable Case

First, navigate to the `svm` directory and open `p1abc.py`. At the top you will notice three boolean variables, `p1a`, `p1b`, `p1c`. For now, please set `p1b` and `p1c` to `False`. Run `p1abc.py`. You should see a figure pop up with a bunch of red data and a bunch of blue data in distinct groups. This is the known and classified data that we will use to train our first SVM.

Now, set both `p1a` and `p1b` to be `True`. This is where we actually train our data.

Within the `p1b` if statement starting around line 46, you will see the following:
```
clf = svm.LinearSCV() # creates a Linear SVC svm class
clf.fit (data, val) # fits the data with and their labels (val)
                    # using a SVM with linear kernel
```
the first command makes `clf` an instance of the LinearSCV class and the second command uses the fit method to generate a support vector that separates the(x,y) data points based on known their value/classification. In this case, if you see in `data_a.xlsx` the red points in the bottom left corner are classified as a `0` and the blue points are classified as a `1`.

After the data has been fitted, the svm is used to predict the classification of two additional test points, plotted with + signs. You should see that they both appear blue, meaning the svm classified those data parts as most likely belonging to the `1` label.

Now, set `p1a`, `p1b` and, `p1c` to `True` at the top and run `p1abc.py`. You should see mostly the same plot should appear but now there is a black line running through the middle of the graph. This black line is the decision boundary determined by the SVM!

| :question: QUESTION 1 :question:   |
|:---------------------------------------------------|
| What do you think might happen if a data point were to fall exactly on the decision boundary? |

### 2.2 Nonlinear SVM

Now, open `p1def.py`. At the top you will notice three boolean variables, `p1d`, `p1e`, `p1f`. For now, please set `p1e` and `p1f` to `False`. Go ahead and run the code. This first section is for data visualization. You should see a figure pop up with a bunch of red data and a bunch of blue data in distinct groups. Now here it can be fairly obvious that a simple line will not separate the data. This is where using different kernels comes in to play!

The full definition of the SVC method with its default values is as follows and can be found [here](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).
```
class sklearn.svm.SVC(*, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, 
        probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, 
        max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)
```

We don't often need to deal with ALL of these values, which is why we start off just using the defaults. In line 47 of the code we simply have:
```
clf = svm.SVC()
```

This means that all of the default parameters are used. Depending on the chosen kernel, only certain parameters are actually used by the method. For example `gamma` is not used if the kernel is linear. Don’t worry too much about this now. Check out the above link if you want to see the definition of the class and learn a little more.

Now, change `p1d` and `p1e` to `True` and run the code again. In this section, instead of using the `svm.LinearSV`C method, we are using the `svm.SVC` method to fit our data. By default, the `svm.SVC` method uses a radial basis function as its kernel and it has two parameters, in this case `gamma` and `C`. The `gamma` parameter defines how far the influence of a single training example reaches, with low values meaning far and high values meaning close. The `gamma` parameters can be seen as the inverse of the radius of influence of samples selected by the model as support vectors.

The `C` parameter trades off correct classification of training examples against maximization of the decision function’s margin. For larger values of `C`, a smaller margin will be accepted if the decision function is better at classifying all training points correctly. A lower `C` will encourage a larger margin, therefore a simpler decision function, at the cost of training accuracy. In other words `C` behaves as a regularization parameter in the SVM.

By default `C` is set to 1 and `gamma` is set to 'scale' which means it uses `1 / (nfeatures * X.var())` as the value of `gamma`.

Here you will see an additional data point indicated by a +. Although the point appears to be directly between the four clusters of data, the SVM has classified it as "blue". Why?

The answer can be revealed by plotting the decision boundaries! Now change `p1d`, `p1e`, and `p1f` to `True` and run the code again. Here you should see the same plot but with the decision boundaries dictated by a solid black line, the margins dictated by dashed lines. If you recall, the goal of SVM is to find a decision boundary that maximizes the margins separating the two data sets. You'll also notice that some points are circled with green lines. These points are the support vectors, basically the points that have the most influence on determining the location of the decision boundary. In this case, the decision boundary connects the two blue sections in the middle, while cutting off the red sections from each other.

Try changing the values of `C` and `gamma`.
```
clf = svm.SVC(C = 1.0, kernel = 'rbf', gamma = 'scale')
```

Try changing the values of C and gamma and see what happens! Show us some
screenshots of any notable changes. Start with gamma values ranging from .1-10 and C values
ranging from .1-100. Feel free to explore other values.

| :question: QUESTION 2 :question:   |
|:---------------------------------------------------|
|  |

Now, lets try changing the kernel and see what happens. Go to the line of code in the p1e IF
statement (for me it is line 47) and change it to the following
clf = svm .SVC ( kernel =’poly ’)
The following kernels are available for use: ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’.

| :question: QUESTION 3 :question:   |
|:---------------------------------------------------|
|  |\
How well does each kernel appear to classify the data? Show some screenshots of the
different kernels being used.
Next, lets just see what happens if we play with the polynomial kernel. By default, it is set to
degree=3. Lets use a higher degree and see if it helps.
clf = svm .SVC ( kernel =’poly ’, degree =4)

| :question: QUESTION 4 :question:   |
|:---------------------------------------------------|
| Does changing the polynomial kernel degree help? Which one appears to be the best?
Is there a disadvantage to using higher degree polynomial functions? |

## 3 Neural Network (NN)

This is the classical ”hello world” example of neural networks used to classify handwritten images
of numbers to the number that’s written. Run the following command:

cd nn
python3 classifier .py

A window will show up, there are two sets of accuracy over ”epochs”, one for the training data
and the other for the testing data. As the epoch proceeds, we can see that the accuracies in both
the training set and the test set increase as expected. Notice this is an ideal case. Over-fitting
could happen if the epoch number is set too high and under-fitting could happen when the number
is too low. Go into the source code and give it an epoch number larger than 5.
Now take a closer look into the code itself. In the previous section we have installed the
Tensorflow library. It is an open source library developed by Google for convenient and efficient
deployment of common machine learning techniques. Keras is the neural network library that is
built on Tensorflow. Some background information: An alternative library is Pytorch, developed
by Microsoft and Facebook, feel free to implement with both libraries and make a comparison. In
2.12 we will stick with Tensorflow.
In this lab we use the MNIST image set, which is a set of handwritten images of numbers that
are correctly labelled. Each image contains 28 × 28 pixels. In this script, we use 60,000 image to
train our network and 10,000 to test the network. Several steps to getting the input data to the
write format. The 28 × 28 pixels are converted to a single array. Then they are fed through the
net, where the trained y value is the number corresponding to the image.
Here we use two layers of neurons, with a ’sigmoid’ and a ’softmax’ activation function. There
are a lot other activation functions, such as ’relu’ and ’tanh’. Give it a try and see how that
changes the result. Here, we use stochastic gradient decent to find our global minimum. Alternative
optimizers such as ’adam’ and ’adagrad’ are also included in Keras. A cool gif of their performance
can be found here: :https://mlfromscratch.com/optimizers-explained/#/
Eventually, we give each image sample 10 scores based on the output of the last layer of neuron.
These 10 scores are the probabilities, corresponding to the numbers 0-9, that are evaluated with a
mathematical technique called cross-entropy. The index with the highest probability is the number
predicted by the image.

| :question: QUESTION 5 :question:   |
|:---------------------------------------------------|
| Why do we convert the 2-dimensional 28-28 input matrix into a 784 x 1 array? How is that different from convolutional neural network? |

| :question: QUESTION 6 :question:   |
|:---------------------------------------------------|
| Do you notice any interesting correlation between sigmoid and softmax as activation functions? |

## 4 Convolutional Neural Network (CNN)

You can run the CNN code on Google Colab. Open this [example notebook](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/segmentation.ipynb) from TensorFlow.

This is an example of more advanced machine learning that would be likely necessary to detect something like an transparent water bottle. In this example, we will segment the images and classify them. We use the Oxford IIIT Pet dataset, where 37 categories of pets are correctly segmented from the image and classified. This specific example is using a modified version of a convolutional neural network, which has several down sampling and then up sampling layers. The down-sampling encoder is called MobileNetV2, details can be found [here](https://arxiv.org/pdf/1801.04381.pdf). The up-sampling decoder is the `pix2pix` package from the Tensorflow example.

The code is done in several parts, the first is displaying the image, and the mask. The mask is the segmented image that we want to produce with the model, and the image is of an animal. The next part is the construction of the net and the fitting of the system. It requires a lot more training than the previous example, 20 epochs instead of 5. After each epoch, there is a callback written to save the weights at the end of the training and also a callback to display the predicted image after training. The last section is the training loss and validation loss after each epoch. There is also a saved model in the folder.

| :question: QUESTION 7 :question:   |
|:---------------------------------------------------|
| At which stage does 'convolution' comes in? |

## 5 Submission and Feedback Form

Please compile all your screenshots and answers in a PDF and upload it to Canvas. Then, fill out https://tinyurl.com/212-feedback.

[^1]: Version 1 - 2020: Jerry Ng, Rachel Hoffman-Bice, Steven Yeung, and Kamal Youcef-Toumi
  Version 2 - 2021: Phillip Daniel
  Version 3 - 2024: Jinger Chong