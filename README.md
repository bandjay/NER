# Named Entity Recognition And Typing
# ===================================

### Entity recognition and typing is an important study due to the many forms of text collections that are emerging, such as mobile applications and social media. Most text data is unstructured and does not provide much meaning or knowledge. Entity recognition and typing will structure the data and provide more meaning about the text data.

### This framework uses distant supervision for typing ,state of the art filtering techniques for detecting quality entity mentions and Autoencoder based anomaly detection to remove noisy seed mentions.

### [[NER Framework Paper](https://github.com/bandjay/NER/blob/master/language-independent-entity-final.pdf)]

### Dependencies

* python 2.7
* numpy, scipy, scikit-learn, lxml, TextBlob and related corpora

```
$ sudo pip install numpy scipy sklearn lxml textblob h2o
$ sudo python -m textblob.download_corpora
```
### Default Run

```
$ ./run.py  
```

###  Change the paths for Raw text, seed file and type file.
###  Additional parameters can be adjusted in run.py

### Examples dataset and reference work can be found at [[ClusType Reference](https://github.com/shanzhenren/ClusType)]
