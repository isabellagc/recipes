import click
import torch
import time
import pandas as pd
import random
from tqdm import tqdm

#-----brought in from example project might delete later
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
from sklearn.model_selection import train_test_split


from sklearn.linear_model import LinearRegression

import json
import re

# Custom defined
# from models import *
#from utils import data_utils, test_utils

# New
from gensim.models import Doc2Vec
from sklearn import utils
#from sklearn.model_selection import train_test_split
import gensim
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

from abc import abstractmethod
#import os
import tempfile
from subprocess import PIPE
import warnings

from six import text_type

from nltk.internals import find_file, find_jar, config_java, java, _java_options
from nltk.tag.api import TaggerI

_stanford_url = 'https://nlp.stanford.edu/software'



class StanfordTagger(TaggerI):
    """
    An interface to Stanford taggers. Subclasses must define:

    - ``_cmd`` property: A property that returns the command that will be
      executed.
    - ``_SEPARATOR``: Class constant that represents that character that
      is used to separate the tokens from their tags.
    - ``_JAR`` file: Class constant that represents the jar file name.
    """

    _SEPARATOR = ''
    _JAR = ''

    def __init__(
        self,
        model_filename,
        path_to_jar=None,
        encoding='utf8',
        verbose=False,
        java_options='-mx1000m',
    ):
        # Raise deprecation warning.
        warnings.warn(
            str(
                "\nThe StanfordTokenizer will "
                "be deprecated in version 3.2.6.\n"
                "Please use \033[91mnltk.parse.corenlp.CoreNLPParser\033[0m instead."
            ),
            DeprecationWarning,
            stacklevel=2,
        )

        if not self._JAR:
            warnings.warn(
                'The StanfordTagger class is not meant to be '
                'instantiated directly. Did you mean '
                'StanfordPOSTagger or StanfordNERTagger?'
            )
        self._stanford_jar = find_jar(
            self._JAR, path_to_jar, searchpath=(), url=_stanford_url, verbose=verbose
        )

        self._stanford_model = find_file(
            model_filename, env_vars=('STANFORD_MODELS',), verbose=verbose
        )

        self._encoding = encoding
        self.java_options = java_options

    @property
    @abstractmethod
    def _cmd(self):
        """
        A property that returns the command that will be executed.
        """

    def tag(self, tokens):
        # This function should return list of tuple rather than list of list
        return sum(self.tag_sents([tokens]), [])


    def tag_sents(self, sentences):
        encoding = self._encoding
        default_options = ' '.join(_java_options)
        config_java(options=self.java_options, verbose=False)

        # Create a temporary input file
        _input_fh, self._input_file_path = tempfile.mkstemp(text=True)

        cmd = list(self._cmd)
        cmd.extend(['-encoding', encoding])

        # Write the actual sentences to the temporary input file
        _input_fh = os.fdopen(_input_fh, 'wb')
        _input = '\n'.join((' '.join(x) for x in sentences))
        if isinstance(_input, text_type) and encoding:
            _input = _input.encode(encoding)
        _input_fh.write(_input)
        _input_fh.close()

        # Run the tagger and get the output
        stanpos_output, _stderr = java(
            cmd, classpath=self._stanford_jar, stdout=PIPE, stderr=PIPE
        )
        stanpos_output = stanpos_output.decode(encoding)

        # Delete the temporary file
        os.unlink(self._input_file_path)

        # Return java configurations to their default values
        config_java(options=default_options, verbose=False)

        return self.parse_output(stanpos_output, sentences)


    def parse_output(self, text, sentences=None):
        # Output the tagged sentences
        tagged_sentences = []
        for tagged_sentence in text.strip().split("\n"):
            sentence = []
            for tagged_word in tagged_sentence.strip().split():
                word_tags = tagged_word.strip().split(self._SEPARATOR)
                sentence.append((''.join(word_tags[:-1]), word_tags[-1]))
            tagged_sentences.append(sentence)
        return tagged_sentences



class StanfordPOSTagger(StanfordTagger):
    """
    A class for pos tagging with Stanford Tagger. The input is the paths to:
     - a model trained on training data
     - (optionally) the path to the stanford tagger jar file. If not specified here,
       then this jar file must be specified in the CLASSPATH envinroment variable.
     - (optionally) the encoding of the training data (default: UTF-8)

    Example:

        >>> from nltk.tag import StanfordPOSTagger
        >>> st = StanfordPOSTagger('english-bidirectional-distsim.tagger')
        >>> st.tag('What is the airspeed of an unladen swallow ?'.split())
        [('What', 'WP'), ('is', 'VBZ'), ('the', 'DT'), ('airspeed', 'NN'), ('of', 'IN'), ('an', 'DT'), ('unladen', 'JJ'), ('swallow', 'VB'), ('?', '.')]
    """

    _SEPARATOR = '_'
    _JAR = 'stanford-postagger.jar'

    def __init__(self, *args, **kwargs):
        super(StanfordPOSTagger, self).__init__(*args, **kwargs)

    @property
    def _cmd(self):
        return [
            'edu.stanford.nlp.tagger.maxent.MaxentTagger',
            '-model',
            self._stanford_model,
            '-textFile',
            self._input_file_path,
            '-tokenize',
            'false',
            '-outputFormatOptions',
            'keepEmptySentences',
        ]
def get_json_recipes():
    print("parsing out the recipes...")
    with open('data/epicurious/full_format_recipes.json', 'r') as f:
        recipe_ingredients_dict = json.load(f)
    return recipe_ingredients_dict
'''
    for i in range(10):
        print(recipe_ingredients_dict[i]['ingredients'][0])
        print('=' * 100)'''

def removeNums(x):
    # sprint x
    result = [re.sub('[^a-zA-Z ]+', '', val) for val in x]
    return result

def run():
    json_recipes = get_json_recipes()
    recipes = pd.DataFrame.from_dict(json_recipes, orient='columns')
    # recipes['ingredients'] = recipes['ingredients'].str.replace('\d+', '')
    recipes = recipes.dropna(subset=['rating', 'ingredients'])
    recipes['ingredients'] = recipes['ingredients'].apply(removeNums)
    #NOW CHANGE FROM VEC OF STRINGS TO ONE FAT STRING
    recipes['ingredients'] = recipes['ingredients'].apply(lambda x : " ".join(str(word) for word in x))
    st = StanfordPOSTagger('english-bidirectional-distsim.tagger')
    #print(recipes['ingredients'])
    #f = lambda x: x.split()
    #recipes['tags'] = recipes['ingredients'].apply(f)
    #print(recipes['tags'])
    print(st.tag(recipes['ingredients']))
    #print(st.tag(recipes['ingredients'].split()))

run()
'''


[docs]class StanfordNERTagger(StanfordTagger):
    """
    A class for Named-Entity Tagging with Stanford Tagger. The input is the paths to:

    - a model trained on training data
    - (optionally) the path to the stanford tagger jar file. If not specified here,
      then this jar file must be specified in the CLASSPATH envinroment variable.
    - (optionally) the encoding of the training data (default: UTF-8)

    Example:

        >>> from nltk.tag import StanfordNERTagger
        >>> st = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz') # doctest: +SKIP
        >>> st.tag('Rami Eid is studying at Stony Brook University in NY'.split()) # doctest: +SKIP
        [('Rami', 'PERSON'), ('Eid', 'PERSON'), ('is', 'O'), ('studying', 'O'),
         ('at', 'O'), ('Stony', 'ORGANIZATION'), ('Brook', 'ORGANIZATION'),
         ('University', 'ORGANIZATION'), ('in', 'O'), ('NY', 'LOCATION')]
    """

    _SEPARATOR = '/'
    _JAR = 'stanford-postagger-2018-10-16/stanford-ner.jar'
    _FORMAT = 'slashTags'

    def __init__(self, *args, **kwargs):
        super(StanfordNERTagger, self).__init__(*args, **kwargs)

    @property
    def _cmd(self):
        # Adding -tokenizerFactory edu.stanford.nlp.process.WhitespaceTokenizer -tokenizerOptions tokenizeNLs=false for not using stanford Tokenizer
        return [
            'edu.stanford.nlp.ie.crf.CRFClassifier',
            '-loadClassifier',
            self._stanford_model,
            '-textFile',
            self._input_file_path,
            '-outputFormat',
            self._FORMAT,
            '-tokenizerFactory',
            'edu.stanford.nlp.process.WhitespaceTokenizer',
            '-tokenizerOptions',
            '\"tokenizeNLs=false\"',
        ]

[docs]    def parse_output(self, text, sentences):
        if self._FORMAT == 'slashTags':
            # Joint together to a big list
            tagged_sentences = []
            for tagged_sentence in text.strip().split("\n"):
                for tagged_word in tagged_sentence.strip().split():
                    word_tags = tagged_word.strip().split(self._SEPARATOR)
                    tagged_sentences.append((''.join(word_tags[:-1]), word_tags[-1]))

            # Separate it according to the input
            result = []
            start = 0
            for sent in sentences:
                result.append(tagged_sentences[start : start + len(sent)])
                start += len(sent)
            return result

        raise NotImplementedError

'''

'''

def setup_module(module):
    from nose import SkipTest

    try:
        st = StanfordPOSTagger('stanford-postagger-2018-10-16/models/english-bidirectional-distsim.tagger',
                               'stanford-postagger-2018-10-16/stanford-postagger.jar')
        print("hello")
        print(st.tag(recipes['ingredients'].split()))
    except LookupError:
        raise SkipTest(
            'Doctests from nltk.tag.stanford are skipped because one \
                       of the stanford jars cannot be found.'
        )'''

'''
import os
java_path = "C:\\Program Files\\Java\\jdk1.8.0_102\\bin\\java.exe"
os.environ['JAVAHOME'] = java_path

from nltk.tag.stanford import StanfordPOSTagger
path_to_model = "stanford-postagger-2018-10-16/models/english-bidirectional-distsim.tagger"
path_to_jar = "stanford-postagger-2018-10-16/stanford-postagger.jar"
tagger=StanfordPOSTagger(path_to_model, path_to_jar)
tagger.java_options='-mx4096m'          ### Setting higher memory limit for long sentences
#sentence = 'This is testing'
#print tagger.tag(sentence.split())

def get_json_recipes():
    print("parsing out the recipes...")
    with open('data/epicurious/full_format_recipes.json', 'r') as f:
        recipe_ingredients_dict = json.load(f)

    for i in range(10):
        print(recipe_ingredients_dict[i]['ingredients'][0])
        print('=' * 100)

    return recipe_ingredients_dict

def vectorize():
    json_recipes = get_json_recipes()
    recipes = pd.DataFrame.from_dict(json_recipes, orient='columns')
    # recipes['ingredients'] = recipes['ingredients'].str.replace('\d+', '')
    recipes = recipes.dropna(subset=['rating', 'ingredients'])
    recipes['ingredients'] = recipes['ingredients'].apply(removeNums)
    #NOW CHANGE FROM VEC OF STRINGS TO ONE FAT STRING
    recipes['ingredients'] = recipes['ingredients'].apply(lambda x : " ".join(str(word) for word in x))
    print(tagger.tag(recipes['ingredients'].split()))'''



    
