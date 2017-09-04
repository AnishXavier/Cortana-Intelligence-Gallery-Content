# Deep Learning for Text Classification in Azure GPU-Accelerated Infrastructure

Natural Language Processing (NLP) is one of the fields in which deep learning has made a significant progress. Specifically, the area of text classification, where the objective is to categorize text or sentences into classes, has attracted the interest of both industry and academia. One of the most interesting applications is sentiment analysis, whose objective is to determine whether the attitude expressed in a piece of text towards a particular topic is positive, negative or neutral. This information can be used by companies to define marketing strategy, generate leads or improve customer service. 

In this notebook we will demonstrate how to use Deep Neural Networks (DNNs) in a text classification problem. We will explain how to generate an end-to-end pipeline to train a DNN for text classification and prepare the model for production, so it can be queried by a user to classify sentences via a web service. 

The tools we will use in this entry are:

* [Azure NC24 VM](https://azure.microsoft.com/en-us/blog/azure-n-series-general-availability-on-december-1/) which contains 4 Tesla K80 GPUs.
* The deep learning library [MXNet](https://github.com/dmlc/mxnet). We implemented the project with the commit `962271410059156180ab1d5e79b805e687512be9`.
* [Microsoft R Server](https://www.microsoft.com/en/server-cloud/products/r-server/default.aspx) version 8.0.5.
* [Anaconda](https://www.continuum.io/downloads) with Python version 3.5.
* [Azure Cloud Services](https://azure.microsoft.com/en-gb/services/cloud-services/).
* [CUDA](https://developer.nvidia.com/cuda-toolkit) version 8.0. 
* [cuDNN](https://developer.nvidia.com/cudnn) version 5.1.
* [Math Kernel Library](https://software.intel.com/en-us/intel-mkl) (MKL) version 11.3

To configure and install the environment please refer to this [blog post](https://blogs.technet.microsoft.com/machinelearning/2016/09/15/building-deep-neural-networks-in-the-cloud-with-azure-gpu-vms-mxnet-and-microsoft-r-server/).

## Architecture

In the next figure we show the complete architecture:
<p align="center">
<img src="https://mxnetstorage.blob.core.windows.net/public/nlp/architecture.png" alt="architecture" width="60%"/>
</p>

As it can be seen in the Figure, the first step is to process the dataset. In most of the situations, the dataset will not fit in the memory of the VM we are using, so we must feed the data to the training process in batches: reading a data chunk, processing it to create a group of encoded images and freeing the memory before starting again. This process is programmed in R. Once the model is trained, we can host it in a VM and use it to classify sentences via a web service. We created a Python API that communicates a simple front end with the classification process. The front end is programmed in JavaScript and HTML, and provides a flexible environment to switch between different trained models.  

## Text Classification at Character Level

The area of text classification has been developed mostly with machine learning models that use features at the word level. The idea of using deep learning for text classification at character level came first in 2015 with the [Crepe model](https://arxiv.org/abs/1509.01626). The following year the technique was developed further in the [VDCNN model](https://arxiv.org/abs/1606.01781) and the [char-CRNN model](https://arxiv.org/abs/1602.00367). We implemented the code of the Crepe model ([R version](./R/crepe_model.R) and [python version](./python/03%20-%20Crepe%20-%20Amazon%20(advc).py)) and the VDCNN model ([R version](./R/vdcnn_model.R) and [python version](./python/04%20-%20VDCNN%20-%20Amazon(advc).py).  

DNNs have achieved good results when used together with raw data, especially in computer vision where the inputs to the network are the pixels of the image, normally without any preprocessing. In an equivalent way, characters are the atomic representation of the sentence. 

The encoding of each sentence is represented in the following figure: 

<p align="center">
<img src="https://mxnetstorage.blob.core.windows.net/public/nlp/matrix_text.png" alt="Text encoding" width="60%"/>
</p>

Each sentence is transformed into a matrix, where the rows corresponds to a dictionary and the columns corresponds to the character in the sentence. The dictionary consists of the following characters:

```
abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}
```

For each character in the sentence, we compute a 1 hot encoding vector, i.e., for each column, we assign a 1 to the corresponding row. As it is usually done when using images with DNNs, the size of the matrix is fixed. Therefore, all sentences are trimmed to a maximum of 1014 characters. In the case the sentence is shorter, the matrix is padded with spaces. At the end, each encoded sentence has a size of 69, which is the length of the vocabulary, times 1014, which is the length of the sentence.

## Convolution with Characters

A convolution allows to generate hierarchical representations mapping from the inputs, to the internal layers, and to the outputs. Each layer sequentially extracts features from small windows in the input sequence and aggregates the information through an activation function. These windows, normally referred as kernels, propagate the local relationships of the data over the hidden layers. 

The next image represents an example of a kernel in an image (left) and in a sentence (right). 

<p align="center">
<img src="https://mxnetstorage.blob.core.windows.net/public/nlp/kernel.png" alt="Kernel in images and sentences" width="60%"/>
</p>

As in can be seen in the figure, a kernel in an image has usually a size of `3x3`, `5x5` or `7x7`. A convolution of 3, 5 or 7 pixels can represent a small part of the image, like an edge or shape. 

In a model like Crepe or VDCNN, the first convolution has the size of `3xn` or `7xn`, where n is the size of the vocabulary, perhaps because the average word is 7 characters. The subsequence convolutions have size of `3x1`, which can be loosely interpreted as a trigram approach. Other options are kernels of `5x1` or `7x1`.

In this entry, we analyze the Crepe model, composed by 9 layers (6 convolutional and 3 fully connected). To compute the model in R you have to type:

    cd R
    Rscript text_classification_cnn.R --network crepe --batch-size 512 --lr 0.01 --gpus 0,1,2,3 --train-dataset categories_train_big.csv --val-dataset categories_test_big.csv --num-classes 7 --num-round 10 --log-dir $PWD --log-file crepe_amazon_categories_mrs.log --model-prefix crepe_amazon_categories_mrs 

As it is shown in the command line, we used a batch size of 512 in 4 GPUs, which means that in every epoch each GPU processes 128 images.

In the next figure we show the results of computing the Crepe model on the Amazon categories dataset. The dataset can be downloaded using [this script](./data/download_amazon_categories.py). 

<p align="center">
<img src="https://mxnetstorage.blob.core.windows.net/public/nlp/accuracy_crepe.png" alt="Accuracy of Crepe DNN in Amazon categories dataset" width="60%"/>
</p>

This dataset consists of a training set of 2.38 million sentences, a test set of 420,000 sentences, divided in 7 categories: “Books”, “Clothing, Shoes & Jewelry”, “Electronics”, “Health & Personal Care”, “Home & Kitchen”, “Movies & TV” and “Sports & Outdoors”. The model has been trained for 10 epochs in an Azure NC24 with 4 K80 Tesla GPUs. The training time was around 1 day.

We also provide several examples written in python. The [first example](./python/01%20-%20LeNet%20-%20MNIST%20Walkthrough.ipynb) shows how to create a custom iterator to train a DNN. The [second example](./python/02%20-%20Crepe%20-%20Amazon.ipynb) explains how to compute the Crepe model in the [Amazon sentiment dataset](https://mxnetstorage.blob.core.windows.net/public/nlp/amazon_review_polarity_csv.tar.gz). This dataset contains a train set of 3.6 million sentences and a test set of 400,000. It has two classes, positive and negative. In the [third example](./python/03%20-%20Crepe%20-%20Amazon%20(advc).py) we show how to train the Crepe model in the same dataset, but this time we feed the data asynchronously using a pre-fetch method. Finally, in the [fourth example](./python/04%20-%20VDCNN%20-%20Amazon(advc).py) we demonstrate how to create the VDCNN architecture and implemented a k-max pooling layer using MXNet API. The training of the Crepe model for 10 epochs in 1 GPU is about 2 days and with the VDCNN model is about 5 days.

## Development of Cloud Infrastructure for Text Classification in Azure

Once we have the DNN trained model, we can use [Azure Web Apps](https://azure.microsoft.com/en-us/services/app-service/web/) to operationalize the solution and provide text classification as a web service. The logic of the application is managed via a python API running on [Flask](http://flask.pocoo.org/docs/0.12/), that gets the sentence as input and returns the classification score. 

The front end is managed by an AngularJS application, that is in charge of sending the information to the server, receiving the response and showing it to the user.

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


In the following section we will show you how to deploy a [simplified version](https://mxnetdeepapi.azurewebsites.net/) of the web app presented above.

### Prerequisites to deploy a MXNet model on Azure Web Apps

- Create an account in [Azure Web Apps](https://azure.microsoft.com/en-us/services/app-service/web/)
- Have the latest `Azure cli` installed and in your `PATH` environment variable.
- Have `git` installed and in your `PATH`.
- Register on the [NVIDIA cuDNN website](https://developer.nvidia.com/cudnn) to be able to use the cuDNN library, and download `cudnn64_70.dll`.
- Download the code of a simplified version of the webapp [here](https://mxnetstorage.blob.core.windows.net/public/nlp/NLPWebApp.zip) and unzip it. It contains the pre-trained model for sentiment analysis. The final result is visible [here](https://mxnetdeepapi.azurewebsites.net/).

### cuDNN

- Add the downloaded `cudnn64_70.dll` to `MXNET\3rdparty\cudnn`
- __zip the MXNet folder__ so that there is a `MXNET.zip` file at the root of your `NLPWebApp` folder containing the content of the `MXNET` folder.

### Create an Azure App Service Website

- On the command line, make sure your working directory is the `NLPWebApp` directory:
```
cd NLPWebApp
```

- First login on your subscription using the Azure CLI in linux or windows:
```
azure login
azure config mode asm
```

- Create an Azure App Service Website (if you don't have `git` in your path you will have an error and you will need to add the remote manually):
```
azure site create --git <your_app_name>
``` 

### Modify your Azure App Service to support 64 bit, and install python 2.7.11 64 bit
- In the portal, navigate to your newly created Azure App Service.
- Update your plan to use at least the B1 plan (we recommend using the B2 plan or the S2 plan to benefit from autoscale).
- In the __Application Settings__ menu, set the app to _64 bits_, and _Always On_.
- In the __Extension__ menu, add the _Python 2.7.11 x64_ extension from Steve Dower. This will install a 64 bit version of Python 2.7.11.

### Deploy your app to the cloud
- Inside the folder of your application run the following command:
```
git push azure master
```

### Test the app
- The application should be running at : `http://<your_app_name>.azurewebsites.net`.
- You can call the API using the `/api/sentiment` endpoint and sending a POST request with the following payload:
```
{
    "sentence": "This is a test for the API"
}
```

### Modifying the app to deploy your own model

#### deploy.cmd
- This is the command file that is run after deployment, you can update it if you need to install a different version of MXNet for example or perform extra post-deployment actions.

#### requirements.txt
- This file list all the pip dependencies necessary for your flask app to run.
- If your dependencies are not pure python, it is recommended to install them via a wheel (see numpy and pandas for example), pre-compiled for the windows 64 bit, python 2.7.

#### web.2.7.config
- This file contains the webconfig for the IIS running on the Azure App Service instances. You can use this one or modify it to use the `fastcgi` protocol if that is more suitable for your use-case.

#### Model/crepe_amazon_sentiment-*
- These files contains the symbol definition of your network and the weights of the model, they are necessary to load a trained MXNet model in memory.
- Replace them with your own files and update the `model.py` file in the WebApp folder.

#### WebApp/model.py
- That's where the model is loaded and the routes are defined, use this file to add more models or more routes to your API.

#### WebApp/index.html
- If you want your API to have a frontend, or documentation, you can define static pages as well, the assets are under `static/assets/*`.
