# Deep Learning for Text Classification in Azure GPU-Accelerated Infrastructure

Natural Language Processing (NLP) is one of the fields in which deep learning has made a significant progress. Specifically, the area of text classification, where the objective is to categorize text or sentences into classes, has attracted the interest of both industry and academia. One of the most interesting applications is sentiment analysis, whose objective is to determine whether the attitude expressed in a piece of text towards a particular topic is positive, negative or neutral. This information can be used by companies to define marketing strategy, generate leads or improve customer service. 

In this notebook we will demonstrate how to use Deep Neural Networks (DNNs) in a text classification problem. We will explain how to generate an end-to-end pipeline to train a DNN for text classification and prepare the model for production so it can be queried by a user to classify sentences via a web service. 

The tools we will use in this entry are:

* [Azure NC24 VM](https://azure.microsoft.com/en-us/blog/azure-n-series-general-availability-on-december-1/) which contains 4 Tesla K80 GPUs.
* The deep learning library [MXNet](https://github.com/dmlc/mxnet). We implemented the project with the commit `962271410059156180ab1d5e79b805e687512be9`.
* [Microsoft R Server](https://www.microsoft.com/en/server-cloud/products/r-server/default.aspx) version 8.0.5.
* [Anaconda](https://www.continuum.io/downloads) with Python version 3.5.
* [Azure Cloud Services](https://azure.microsoft.com/en-gb/services/cloud-services/).
* [CUDA](https://developer.nvidia.com/cuda-toolkit) version 8.0. 
* [CuDNN](https://developer.nvidia.com/cudnn) version 5.1
* [Math Kernel Library](https://software.intel.com/en-us/intel-mkl) (MKL) version 11.3

To configure and install the environment please refer to this [blog post](https://blogs.technet.microsoft.com/machinelearning/2016/09/15/building-deep-neural-networks-in-the-cloud-with-azure-gpu-vms-mxnet-and-microsoft-r-server/).

## Architecture

In the next figure we show the complete architecture:
<p align="center">
<img src="https://mxnetstorage.blob.core.windows.net/public/nlp/architecture.png" alt="architecture" width="60%"/>
</p>

As it can be seen in the Figure, the first step is to process the dataset. In most of the situations, the dataset will not fit in the memory of the VM we are using, so we must feed the data to the training process in batches: reading a data chuck, processing it to create a group of encoded images and freeing the memory before starting again. This process is programmed in R. Once the model is trained, we can host it in a VM and use it to classify sentences via a web service. We created a Python API that communicates a simple front end with the classification process. The front end is programmed in JavaScript and HTML, and provides a flexible environment to switch between different trained models.  

## Text Classification at Character Level

The area of text classification has been developed mostly with machine learning models that use features at the word level. The idea of using deep learning for text classification at character level came first in 2015 with the [Crepe model](https://arxiv.org/abs/1509.01626). The following year the technique was developed further in the [VDCNN model](https://arxiv.org/abs/1606.01781) and the char-CRNN model. We implemented the code of the [Crepe model](./R/crepe_model.R) and the [VDCNN model](./R/vdcnn_model.R).  

DNNs have achieved good results when used together with raw data, especially in computer vision where the inputs to the network are the pixels of the image, normally without any preprocessing. In an equivalent way, characters are the atomic representation of the sentence. 

The encoding of each sentence is represented in the following figure. 

<p align="center">
<img src="https://mxnetstorage.blob.core.windows.net/public/nlp/matrix_text.png" alt="Text encoding" width="60%"/>
</p>

Each sentence is transformed into a matrix, where the rows corresponds to a dictionary and the columns corresponds to the character in the sentence. The dictionary consists of the following characters:

`abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}`

For each character in the sentence, we compute a 1 hot encoding, i.e., for each column, we assign a 1 to the corresponding row. As it is usually done when using images with DNNs, the size of the matrix is fixed. Therefore, all sentences are trimmed to a maximum of 1014 characters. In the case the sentence is shorter, the matrix is padded with spaces. At the end, each encoded sentence has a size of 69, which is the length of the vocabulary, times 1014, which is the length of the sentence.

# Convolution with Characters

A convolution allows to generate hierarchical representations mapping from the inputs, to the internal layers, and to the outputs. Each layer sequentially extracts features from small windows of the input sequence and aggregates the information through an activation function. This windows, normally referred as kernels, propagates the local relationships of the data over the hidden layers. 

The next image represents an example of a kernel in an image and in a sentence. 

<p align="center">
<img src="https://mxnetstorage.blob.core.windows.net/public/nlp/kernel.png" alt="Kernel in images and sentences" width="60%"/>
</p>

As in can be seen in the figure, a kernel in an image has usually a size of `3x3`, `5x5` or `7x7`. A convolution of 3, 5 or 7 pixels can represent a small part of the image, like an edge or shape. 

In a model like Crepe or VDCNN, the first convolution has the size of `3xn` or `7xn`, where n is the size of the vocabulary, perhaps because the average word is 7 characters. The subsequence convolutions have size of `3x1`, `5x1` or `7x1`, which can be loosely interpreted as a trigram approach. 

In this entry, we analyze the Crepe model, composed by 9 layers (6 convolutional and 3 fully connected). To compute the model in R you have to type:

    cd R
    Rscript text_classification_cnn.R --network crepe --batch-size 512 --lr 0.01 --gpus 0,1,2,3 --train-dataset categories_train_big.csv --val-dataset categories_test_big.csv --num-classes 7 --num-round 10 --log-dir $PWD --log-file crepe_amazon_categories_mrs.log --model-prefix crepe_amazon_categories_mrs 

As it is shown in the command line, we used a batch size of 512 in 4 GPUs, which means that in every epoch each GPU processes 128 images.

In the next figure we show the results of computing the Crepe model on the Amazon categories dataset. The dataset can be downloaded using [this script](./data/download_amazon_categories.py). 

<p align="center">
<img src="https://mxnetstorage.blob.core.windows.net/public/nlp/accuracy_crepe.png" alt="Accuracy of Crepe DNN in Amazon categories dataset" width="60%"/>
</p>

This dataset consists of a training set of 2.38 million sentences, a test set of 420,000 sentences, divided in 7 categories: “Books”, “Clothing, Shoes & Jewelry”, “Electronics”, “Health & Personal Care”, “Home & Kitchen”, “Movies & TV” and “Sports & Outdoors”. The model has been trained for 10 epochs in an Azure NC24 with 4 K80 Tesla GPUs. The training time was around 1 day.

## Development of Cloud Infrastructure for Text Classification in Azure

Once we have the DNN trained model, we can use the Azure cloud infrastructure to operationalize the solution and provide text classification as a web service. The logic of the application is managed via a python service in the server, that gets the sentence as input and returns the classification score. 

The front end is managed by an AngularJS application, that is in charge of sending the information to the server, receiving the response and showing it to the user. For managing the communications between the front and back end, we used a API programmed in Flask. 

We include the Crepe model for text classification, using the Amazon categories dataset and the same architecture trained for sentiment analysis, using the Amazon polarity dataset.

In this [web page](http://osdcwebappdeeplearning.azurewebsites.net/) we show the complete system. Apart from the models we created for this post,  we also included the [text analytics API](https://gallery.cortanaintelligence.com/MachineLearningAPI/Text-Analytics-2) of Cortana Intelligence Suite. 

The next figure shows the result of the text classification API when we introduce in the text box the following review: “It was a breeze to configure and worked straight away.”

<p align="center">
<img src="https://mxnetstorage.blob.core.windows.net/public/nlp/webapp_categories.png" alt="Web app for text classification" width="60%"/>
</p>

From the sentence, we can guess that the user is talking about the setup of some technological device. The system predicts that the highest probability is in the class “Electronics”.

Let’s analyze now some results for the sentiment analysis API and see what happens if we introduce the following sentence in the text box: “It arrived as expected. No complaint.” The result is showed in the following figure. 

<p align="center">
<img src="https://mxnetstorage.blob.core.windows.net/public/nlp/webapp_sentiment.png" alt="Web app for text classification" width="60%"/>
</p>

The Crepe model shows a very positive sentiment while Cortana shows a neutral sentiment. This makes sense if we understand the context of each model. The Crepe model was trained in the Amazon Polarity dataset. In that scenario, if a product arrives as expected, we can state that it is a positive situation. On the contrary, Cortana text API was trained with a different dataset, and probably tuned to cope with broader situations. In a general scenario, something arriving as expected could be seen as a neutral situation.  These results highlight the fact that the result of a machine learning model is closely related to the context in which it has been trained.


You can download the code of the Web app [here](!https://mxnetstorage.blob.core.windows.net/public/nlp/NLPWebApp.zip). It contains the pre-trained model for sentiment analysis and category classification. You will need to download the cudnn DLL library from the [nvidia website](!https://developer.nvidia.com/cudnn) and put it in `MXNET/3rdparty/cudnn`. You will need to put your own API key for the Microsoft data market in `FlaskWebProject/model.py:168`. You can then deploy the code to [Azure Web Apps](!https://azure.microsoft.com/en-us/services/app-service/web/), provided your instance has Python 2.7 (64 bit) installed and at least 2GB of RAM. 
