!pip install konlpy
!pip install progress-timer
import nltk
nltk.download('stopwords')

!pip install cupy
conda install -c anaconda cupy

!wget "http://nlp.stanford.edu/data/glove.6B.zip"
!mkdir pretrained_models
!mv "glove.6B.zip" "pretrained_models/glove.6B.zip"
!unzip "pretrained_models/glove.6B.zip" -d 'pretrained_models'

!wget "http://nlp.stanford.edu/data/glove.840B.300d.zip"
!mv "glove.840B.300d.zip" "pretrained_models/glove.840B.300d.zip"
!unzip "pretrained_models/glove.840B.300d.zip" -d 'pretrained_models'
