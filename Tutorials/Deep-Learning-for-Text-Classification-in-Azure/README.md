# Deep Learning for Text Classification in Azure GPU-Accelerated Infrastructure

Natural Language Processing (NLP) is one of the fields in which deep learning has made a significant progress. Specifically, the area of text classification, where the objective is to categorize text or sentences into classes, has attracted the interest of both industry and academia. One of the most interesting applications is sentiment analysis, whose objective is to determine whether the attitude expressed in a piece of text towards a particular topic is positive, negative or neutral. This information can be used by companies to define marketing strategy, generate leads or improve customer service. 

In this notebook we will demonstrate how to use Deep Neural Networks (DNNs) in a text classification problem. We will explain how to generate an end-to-end pipeline to train a DNN for text classification and prepare the model for production so it can be queried by a user to classify sentences via a web service. 

The tools we will use in this entry are:

* [Azure NC24 VM](https://azure.microsoft.com/en-us/blog/azure-n-series-general-availability-on-december-1/) which contains 4 Tesla K80 GPUs.
* The deep library [MXNet](https://github.com/dmlc/mxnet). We implemented the project with the commit ``.
* [Microsoft R Server](https://www.microsoft.com/en/server-cloud/products/r-server/default.aspx) version 8.0.5.
* [Anaconda](https://www.continuum.io/downloads) with Python version 3.5.
* [Azure Cloud Services](https://azure.microsoft.com/en-gb/services/cloud-services/).
* [CUDA](https://developer.nvidia.com/cuda-toolkit) version 8.0. 
* [CuDNN](https://developer.nvidia.com/cudnn) version 5.1
* [Math Kernel Library](https://software.intel.com/en-us/intel-mkl) (MKL) version 11.3

To configure and install the environment please refer to this [blog post](https://blogs.technet.microsoft.com/machinelearning/2016/09/15/building-deep-neural-networks-in-the-cloud-with-azure-gpu-vms-mxnet-and-microsoft-r-server/).

