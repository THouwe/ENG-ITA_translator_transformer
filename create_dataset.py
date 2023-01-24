#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def format_dataset(eng, ita):
    """Take an English and a Italian sentence pair, convert into input and target.
    The input is a dict with keys `encoder_inputs` and `decoder_inputs`, each
    is a vector, corresponding to English and Italian sentences respectively.
    The target is also vector of the Italian sentence, advanced by 1 token. All
    vector are in the same length.
 
    The output will be used for training the transformer model. In the model we
    will create, the input tensors are named `encoder_inputs` and `decoder_inputs`
    which should be matched to the keys in the dictionary for the source part
    """
    eng = eng_vectorizer(eng)
    ita = ita_vectorizer(ita)
    source = {"encoder_inputs": eng,
              "decoder_inputs": ita[:, :-1]} # between the [start] and [end] signals
    target = ita[:, 1:] # between the [start] and [end] signals
    return (source, target)
  

def make_dataset(pairs, batch_size=64):
    """Create TensorFlow Dataset for the sentence pairs"""
    import tensorflow as tf
    
    # aggregate sentences using zip(*pairs)
    eng_texts, ita_texts = zip(*pairs)
    # convert them into list, and then create tensors
    # tf.random.set_seed(0) # for reproducibility
    dataset = tf.data.Dataset.from_tensor_slices((list(eng_texts), list(ita_texts)))
    return dataset.shuffle(2048)                   .batch(batch_size).map(format_dataset)                   .prefetch(16).cache()

