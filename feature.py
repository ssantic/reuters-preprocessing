#!/usr/local/python-2.7.5/bin/python

""" feature3.py
    -----------
    @author = Ankai Lou
"""

import os
import sys
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import operator

###############################################################################
########## global variables for single-point of control over change ###########
###############################################################################

datafile = 'dataset.csv'

###############################################################################
############### function for printing dataset to .csv document ################
###############################################################################

def generate_csv(documents, features, weights):
    """ function: generate_csv
        ----------------------
        print feature vectors & class labels to .csv file

        :param documents: dictionary of document objects
        :param features: sorted list of features to represent
    """
    dataset = open(datafile, "wb")
    dataset.write('id\t'.encode('ascii', 'ignore'))
    for feature in features:
        dataset.write(feature.encode('ascii', 'ignore'))
        dataset.write('\t'.encode('ascii', 'ignore'))
    dataset.write('class-label:topics\t'.encode('ascii', 'ignore'))
    dataset.write('\n'.encode('ascii', 'ignore'))
    # feature vector for each document
    for i, document in enumerate(documents):
        # document id number
        dataset.write(str(i).encode('ascii', 'ignore'))
        dataset.write('\t'.encode('ascii', 'ignore'))
        # each tf-idf score
        for feature in features:
            dataset.write(str(weights[i][feature]).encode('ascii', 'ignore'))
            dataset.write('\t'.encode('ascii', 'ignore'))
        # topics/places class labels
        dataset.write(str(document['topics']).encode('ascii', 'ignore'))
        dataset.write('\n'.encode('ascii', 'ignore'))
    dataset.close()

###############################################################################
###################### function(s) for feature selection ######################
###############################################################################

def select_features(weights):
    """ function: select_features
        -------------------------
        generated reduced feature list for vector generation

        :param weights: dictionary from results of the tf-idf calculations
        :returns: sorted list of terms representing the selected features
    """
    features = set()
    for doc, doc_dict in weights.items():
        top = dict(sorted(doc_dict.items(), key=operator.itemgetter(1), reverse=True)[:5])
        for term, score in top.items():
            if score > 0.0:
                features.add(term)
    # sort set into list
    return sorted(features)

###############################################################################
############## function(s) for generating weighted tf-idf scores ##############
###############################################################################

def generate_weights(documents):
    """ function: generate_weights
        --------------------------
        perform tf-idf to generate importance scores for words in documents

        :param document: list of documents to use in calculations
        :returns: dictionary of dictionaries: {"id_" : {"word" : score,...}}
    """
    # generate dict for sklearn
    token_dict = dict([])
    for i, document in enumerate(documents):
        token_dict[i] = b' '.join(document['words']['title'] + document['words']['body'])
    # scikit-learn tfidf
    tfidf = TfidfVectorizer()
    weights = tfidf.fit_transform(token_dict.values())
    features = tfidf.get_feature_names()
    return features, weights

###############################################################################
################ main function for generating refined dataset #################
###############################################################################

def generate_dataset(documents, lexicon):
    """ function: generate_dataset
        --------------------------
        select features from @lexicon for feature vectors
        generate dataset of feature vectors for @documents

        :param documents: list of well-formatted, processable documents
        :param lexicon:   list of word stems for selecting features
    """
    print(f'\nGenerating dataset {datafile}')
    words, weights = generate_weights(documents)

    # generate dictionary for feature selection
    weight_array = weights.toarray()
    weight_dict = dict([])
    for i, row in enumerate(weight_array):
        weight_dict[i] = dict([])
        for j, word in enumerate(words):
            weight_dict[i][word] = weight_array[i][j]

    # generate feature list
    print(f'Selecting features for the feature vectors {datafile}')
    features = select_features(weight_dict)

    # write vectors to dataset1.csv
    print(f'Writing feature vector data {datafile}')
    generate_csv(documents, features, weight_dict)
    print(f'Finished generating dataset {datafile}')
