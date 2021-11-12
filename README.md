# CNN-from-scratch
Implementation of a multi-layer fully connected Convolutional Neural Network (CNN) classifier.

## 1. Neural Network (without convolutions)

For the data pre-processing, I have done three things:
1-	Normalizing the input data by dividing each of the RGB values by 255.
X_TRAIN = X_TRAIN.astype(float) / 255.
X_TEST = X_TEST.astype(float) / 255.

2-	Subtracting the mean from all data points in order to zero-center the data.
X_TRAIN -= np.mean(X_TRAIN, axis=0)
X_TEST -= np.mean(X_TEST, axis=0)

3-	Divided all data points by their standard deviation to minimize the spacing.
X_TRAIN /= np.std(X_TRAIN, axis=0)
X_TEST /= np.std(X_TEST, axis=0)

### Choosing the network architecture:
For the number of layers, I have tried using from 1 to 10 layers and I found that applying more than two layers causes the model to over-fit very quickly with no virtual increase in the validation or test accuracies. So, after trying different architectures, I chose two layers, each with 200 nodes.

### Tuning hyper-parameters:
I have used a guided search approach by first starting with small learning rate (10-6) that makes the loss go down. That made the loss go down, but it was virtually not changing (meaning that learning rate has to be increased). After increasing the learning rate to make the loss explode (with very high learning rate), I tuned the random search range to be between 10-1 and 10-3 with big search range (100 loops, each with a different rate tried with 10 epochs) and finally I chose the learning rate that corresponds to the highest validation accuracy.

<p align="center">
<img src="hyper_parameters.png">
</p>

### Training and validation losses
As can be seen from the graphs below, I stopped the training after epoch 200 because the training and validation accuracies are not increasing anymore.  This also saturates at a training accuracies that are in the 90s range and validation accuracies that are in the 40s range which passes the sanity checks (since validation should be close to 1 and validation should be close to 0.5).

<p align="center">
<img src="train_val_loss.png">
</p>

### Comparison to KNN classifier

K-NN classifier:

<p align="center">
<img src="knn_compare.png">
</p>

Neural Network:

<p align="center">
<img src="nn_compare.png">
</p>

As can be shown from the screenshots above, the performance of the neural network is much better than the K-NN in almost all categories. 

### Average Correct Classification Rate (ACCR)

<p align="center">
<img src="accr.png">
</p>

























