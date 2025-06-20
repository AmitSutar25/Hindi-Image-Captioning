# Hindi-Image-Captioning
This project presents a deep learning approach to Hindi image captioning, an innovative intersection of natural language processing and computer vision.
Captioning images in Hindi is the process of using words to describe a picture in Hindi. It is at the crossroads between natural language processing and computer vision, demanding identification of objects in an image as well as with generation of properly formed sentences that are contextually correct. Despite marked strides in English image captioning, Hindi has had very limited research. This knowledge gap is a hindrance to building inclusive technological solutions for non-English speaking populations. The paper focuses on narrowing the above discussed gap by developing deep learning based framework for generating Hindi captions for images. We present state-of-the-art methods, evaluate their performance and discuss nuances and challenges specific to Hindi. By concentrating on Hindi we hope to contribute towards making AI more democratic by ensuring that developments in image captioning also benefit more than just a few people. Further, this paper has implications beyond Hindi; it can be used as a guide for similar studies involving other less documented languages in future studies.

Methodology-

Dataset Preparation

This procedure of preparing data sets is essential for accomplishing Hindi image captioning mechanism. We have used the Flicker 8K dataset that contains 8,000 images and their corresponding English captions. We translate the captions into Hindi manually to form the dataset which should be accurate as well as contextually relevant. Each image in the dataset is processed using BLIP processor while extracting high-dimensional features necessary for generating captions.

Image Feature Extraction

Features are extracted from images using BLIP Processor. Each image is loaded, converted from any format to RGB and preprocessed to fit input requirements of the BLIP model. This processed image then passes through the BLIP model, which gives a high-dimensional feature vector representing key characteristics of an image. The feature vectors are kept in a pickle file so that they can be loaded with ease during training of models.

![image](https://github.com/user-attachments/assets/b2e3d063-ce1a-4bc0-ad65-9f1f3237c648)

Caption Preprocessing

To ensure consistency and facilitate efficient training of the model, the captions were preprocessed by converting text to lower case, removing special characters and adding tokens at their start and end sequences respectively. Also, TensorFlow Keras tokenizer was used in tokenizing these captions while vocabulary size and maximum caption length were established simultaneously.
 
Model Architecture

The current model setup features an encoder-decoder approach, which is meant to work well with the intricacies of Hindi captions. the encoder encodes the image features from the features extracted by the BLIP processor while the decoder produces real captions in the corresponding language using the LSTM based sequence generation model.

![image](https://github.com/user-attachments/assets/0dab61fe-2d4f-4bde-9049-73668a6e1c87)

Encoder

The encoder receives the high-dimensional vectors of the images’ features and first goes through a dropout layer and then a dense layer that uses ReLU as its activation function. This is followed by reducing the dimensionality of the feature vectors where these features are shaped and formatted for decoding.

Decoder

A final caption is produced by an embedding layer, LSTM layers, and dense layers in decoder. Given tokenized captions, embedding layers reduce them to dense vectors that are processed in LSTM layers to capture sequential dependencies. The output of LSTM is summed up with encoded image features through residual connection. Subsequently, dense layers with ReLU and softmax activation functions generate the captions.

Training Process
Several steps are taken during training to make sure that the model learns well from data and performs effectively on new images. For 100 epochs, we train it using a batch size of 32 while saving the best performing models at each checkpoint. A generator function makes it possible for processing batches efficiently during training so that the model constantly gets some data.
 
Data Generator

The data generator function yields batches of input-output pairs, where each input consists of image features and partial captions, and each output is the next word in the caption sequence. This process is crucial for training the model to predict the next word in a caption given the previous words and the image context.

![image](https://github.com/user-attachments/assets/e23ec778-2929-4e5a-9be1-339e5319976e)

Training Loop

The training loop consequently occurs with the rounds of the specified epochs, using the data generator to feed batches of data. The checkpoints are employed to write out the model’s weight parameters when the best validation loss is achieved, in order to keep the best solution.

Evaluation

BLEU scores are used for evaluating how well the model works in machine translation as well as captioning tasks. We get BLEU-4, 3, 2 and 1 scores to evaluate the relevance of text generated by a computer system. Besides this, qualitative analysis is carried out by comparing generated captions against reference captions and attention maps are visualized to show if feature extraction and caption generation processes are effective.

![image](https://github.com/user-attachments/assets/8ad9e0c6-7f07-4472-a43a-2655b0a1613f)

