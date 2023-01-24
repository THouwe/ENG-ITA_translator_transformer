# English-to-Italian Translator: a Transformer Model


## Introduction.
In natural language processing (NLP), **transformer models** are a type of neural network (NN) architecture that have been used to achieve state-of-the-art performance on a variety of tasks, such as machine translation, text summarization, and question answering. Transformers are an attention-based encoder-decoder NN architecture. A transformer model takes as input a sequence of words (for example, a sentence or a document), and produces a new, transformed sequence of words as output (for example, a translated sentence or a summary of the input). The key innovation in transformer models is the **attention** mechanism, which allows the model to weigh the importance of different parts of the input when producing the output. The attention mechanism is a form of memory, which the model uses to "remember" what it has seen so far in the input, and use this information to inform its predictions for the remaining part of the input. Specifically, we will implement a transformer model using a Multi-Headed Attention mechanism, where multiple attention mechanism heads are used in the same layer to improve the performance by allowing the model to attend to different part of the sequence simultaneously. In simple terms, Transformer models are neural networks that are good at understanding sequences of data (such as sentences or paragraphs) by paying attention to different parts of the input and "remembering" what it has seen so far, which helps it make better predictions about what comes next.

Translator models work by learning to map a sentence in one language (the *source language*) to a corresponding sentence in another language (the *target language*). The transformer model is trained on a large dataset of sentence pairs in the source and target languages. During training, the model is presented with a *source sentence*, and its task is to predict the corresponding target sentence. The transformer model works by *encoding* the source sentence into a fixed-length vector representation, called the **context vector**, which contains information about the meaning of the entire sentence. The model then *decodes* the context vector into a target sentence by generating words one at a time. The model is trained to predict the n-th word in the target language from the source sentence (which became the context vector) and the words in the target sentence up to word n-1.




### The Project at a Glance.
In this project we will build from scratch and train an English-to-Italian attention-based transformer translator. Not only do transformer models represent the state-of-the art in machine translation and other NLP tasks, but this project also provides the opportunity of discussing a huge set of concepts relevant for NLP. We will dive deep in many of these concepts and provide references for the curious reader to expand on related topics. To this end, this script can be considered a syllabus of recent NLP techniques. Specifically, the following topics will be covered.




### Index.
The notebook is divided into the following sections:

- [Section 01](Section_01.ipynb): Architecture of the Transformer

- [Section 02](Section_02.ipynb): Data Exploration & Text Normalization (encoding, regular expressions, text normalization)

- [Section 03](Section_03.ipynb): Vectorization (context vector, vectorization layer)

- [Section 04](Section_04.ipynb): Positional Encoding Matrix (sinusoidal positional encoding)

- [Section 05](Section_05.ipynb): Positional Encoding Layer

- [Section 06](Section_06.ipynb): Transformer Building Blocks (attention function, self-attention, cross-attention, feed-forward layers, residual connections)

- [Section 07](Section_07.ipynb): Transformer Encoder and Decoder

- [Section 08](Section_08.ipynb): Building the Transformer

- [Section 09](Section_09.ipynb): Preparing the Transformer Model for Training (masked accuracy, masked loss, learning rate schedule)

- [Section 10](Section_10.ipynb): Training the Transformer (Adam optimizer, dropout rate, early stopping, checkpoint callback)

- [Section 11](Section_11.ipynb): Inference from the Transformer Model

- [Section 12](Section_12.ipynb): Improving the Model (parameter tuning)

### Summary of the Project.

I was able to re-create the transformer architecture described in [Vaswani et al. (2017)](https://arxiv.org/pdf/1706.03762.pdf):

<img src="/Vaswani_et_al_2017_Fig_1.JPG" width="500" height="600"> 



Specifically, I implemented the transformer as follows:

<img src="/section08_figure01_transformer.png" width="500" height="700">



Following training, I obtained a masked accuracy score (probability of predicting the n-th target word) in the validation set of 83%. The result is not bad. Nevertheless, there is a lot of room for improvement. Here are some examples of translation in action:

<img src="/section11_translated_01-to-05.JPG" width="300" height="300">

<img src="/section11_translated_06-to-10.JPG" width="300" height="300">

<img src="/section11_translated_11-to-15.JPG" width="300" height="300">

<img src="/section11_translated_16-to-20.JPG" width="300" height="300">

I am now fine-tuning the parameters. Keep your eyes open for future developments!



### Working Environment.

This tutorial was created using Python 3.9.15 and TensorFlow 2.10.




### Acknowledgements.
The transformer model's architecture is identical to the one described in the seminal article ["Attention Is All You Need" (Google; 2017)](https://arxiv.org/pdf/1706.03762.pdf). Some code snippets are taken by [this tutorial](https://keras.io/examples/nlp/neural_machine_translation_with_transformer/) by François Chollet, and by [this tutorial](https://machinelearningmastery.com/building-transformer-models-with-attention-crash-course-build-a-neural-machine-translator-in-12-days/?utm_source=drip&utm_medium=email&utm_campaign=Build+a+neural+machine+translator+in+12+Days&utm_content=Build+a+neural+machine+translator+in+12+Days) by Adrian Tam. 