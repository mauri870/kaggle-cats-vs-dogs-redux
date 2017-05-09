# Kaggle Cats vs Dogs Redux

Hi, this is my solution for the [dogs vs cats redux kaggle competition](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition) that achieve the 51st place on the oficial public [leaderboard](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/leaderboard). The solution is quite simple because I made it three days until the end of the competition and cannot improve it.

You can find the 3rd position solution on the [kaggle blog](http://blog.kaggle.com/2017/04/20/dogs-vs-cats-redux-playground-competition-3rd-place-interview-marco-lugo/)

# Architecture

I decide to start with a pre-trained model for this competition and I fallback to the Inception V3 model. The model achieve 99.8% accuracy. The final submission score can be improved in several ways, like emsembling more models, xgboost to combine classifiers, preprocess the training images(flip, rotate, scale), use external data, among others.

# Installation

```bash
git clone https://github.com/mauri870/kaggle-cats-vs-dogs-redux.git
cd kaggle-cats-vs-dogs-redux
```

## Preprocessing

First you need to download the train and test data from kaggle. The test and train images must be inside a test and train folder respectively

Run the preprocess script to prepare the data. It'll fit each image in a 299x299 box and fill the blank space in black color.

```bash
go run preprocess.go utils.go
```

Since the inception model expects the train images to be organized into subfolders, let's do that:

```bash
mkdir -p images/{dogs,cats}
cp -v images/train/cat* images/cats
cp -v images/train/dog* images/dogs
```

## Retrain Inception V3

Here's my instructions to build and retrain the inception model:

Let's download and configure tensorflow:

```bash
export TF_VERSION=v1.0.0
wget -qO- https://github.com/tensorflow/tensorflow/archive/${TF_VERSION}.tar.gz | tar zx
cd tensorflow-${TF_VERSION}
./configure
```

Now we will retrain the last fully connected layers of the inception model:

```bash
python tensorflow/examples/image_retraining/retrain.py --flip_left_right --image_dir=$OLDPWD/images
```

Next we need to optimize our model because some ops used to train the original model are now deprecated and in case of the Golang tensorflow bindings will result in a fatal error

```bash
bazel build tensorflow/python/tools/optimize_for_inference
bazel-bin/tensorflow/python/tools/optimize_for_inference --input=/tmp/output_graph.pb --output=/tmp/output_graph_optimized.pb  --frozen_graph=True --input_names=Mul --output_names=final_result
```

## Submission

Now we are ready to create the submission file!

> Note: Refer to the [official page](https://www.tensorflow.org/versions/master/install/install_go) in order to install and configure tensorflow for go

```bash
go run submission.go utils.go
```
