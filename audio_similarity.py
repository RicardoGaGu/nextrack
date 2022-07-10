import librosa
import essentia
import essentia.standard
import essentia.streaming as ess
import pyloudnorm as pyln
import time
import camelotWheel
import pandas as pd
import io
import numpy as np
import streamlit as st
import audiomentations
from matplotlib import pyplot as plt
import librosa.display
from scipy.io import wavfile
import wave
import pydub
import scipy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import ast


class SimilarityEngine:

    def __init__(self,libraryFeaturesPath):
        self.LibraryFeaturesPath = libraryFeaturesPath
        self.libraryFeatures = None
        self.numericFeatures = None
        self.features = None
        self.topK = 5

    # def set_queryPath(self, queryFeaturesPath):
    #     """
    #      Every time the user uploads a new query track
    #     """
    #     self.queryFeaturesPath = queryFeaturesPath

    # def set_libraryPath(self, queryFeaturesPath):
    #     """
    #      Every time the user uploads a new analyzed library of tracks
    #     """
    #     self.libraryFeaturesPath = libraryFeaturesPath

    # def normalizeSubGenreTags(subGenreTags):
    #     "Transform to probabilities the vector of subGenreTags"
    #     normalizer = 1 / float( sum(subGenreTags) )
    #     normalizedsubGenreTags = [x * normalizer for x in subGenreTags]
    #     return normalizedsubGenreTags

    def getEnergyLevels(self):
        classesToLabels = {'low': 1, 'high': 3, 'medium': 2, 'variable': 2}
        
        energyLevels = self.libraryFeatures['energyLevel']
        # Map levels
        energyLevels = energyLevels.map(classesToLabels)
        # Normalization
        energyLevels = self.normalizeFeatures(energyLevels)
        # Add to cleanFeatures df
        self.numericFeatures['energyLevel'] = energyLevels

    def getTempoFeature(self):
        """
        Get tempo features from feature library and do normalization.
        """
        BPM = self.libraryFeatures['bpmRangeAdjusted']
        # Normalize BPM
        BPM = self.normalizeFeatures(BPM)
        # Add to cleanFeatures df
        self.numericFeatures['bpmRangeAdjusted'] = BPM

    def getSubGenreTags(self):
        """ 
        Get subgenre features from feature library.
        """
        subGenres = self.libraryFeatures['subgenre']
        # for subgenre in subGenres.values:
        #     print(type(ast.literal_eval(subgenre)))
        #     print(subgenre)
        f = lambda x : np.array([_value for _key, _value in ast.literal_eval(x).items()])
        subGenres = subGenres.map(f)
        self.numericFeatures['subgenre'] = subGenres

    def getMoodTags(self):
        """ 
        Get mood features from feature library.
        """
        moodTags = self.libraryFeatures['mood']
        f = lambda x : np.array([_value for _key, _value in ast.literal_eval(x).items()])
        moodTags = moodTags.map(f)
        self.numericFeatures['mood'] = moodTags

    def normalizeFeatures(self, feature_column):
        """
        Normalize feature values to [0, 1] using the min-max feature scaling
        """
        scaler = MinMaxScaler()
        return scaler.fit_transform(np.array(feature_column).reshape(-1,1))

        # scaler = StandardScaler()
        # scaler.fit(self.libraryFeatures)
        # normalizedFeatures = scaler.transform(features)
        # return normalizedFeatures

    def compute_distance(self, queryFeatures, recordFeatures):

        """ Two numpy arrays: queryFeatures and libraryFeatures
        Returns similarity value
        """
        # queryFeatures = np.array(queryFeatures)
        # recordFeatures = np.array(recordFeatures)
        # normQueryFeatures = self.normalizeFeatures(queryFeatures)
        # normRecordFeatures = self.normalizeFeatures(recordFeatures)
        try:
            distance = scipy.spatial.distance.cosine(queryFeatures, recordFeatures)
        except:
            distance = 2.
        return distance

    def createLibraryFeatures(self):
        """
        Schema for the features : [Tempo,energyLevels(4),moodVector(X),subGenreTags(Y)]
        """
        # features_df = pd.read_csv(self.libraryFeaturesPath)
        # libraryFeatures = []
        # for recordName in features_df['Record Name']:
        #     libraryFeatures.append([self.getTempoFeature(recordName)].extend(self.getCyaniteFeatures(recordName)))
        # return libraryFeatures

        self.features = pd.DataFrame(columns=['title', 'featureVector']) 
        self.features['title'] = self.numericFeatures['title']
        for i, row in self.numericFeatures.iterrows():
            feature_list = []
            feature_list.append(row['energyLevel'])
            feature_list.append(row['bpmRangeAdjusted'])
            feature_list += list(row['mood'])
            feature_list += list(row['subgenre'])
            self.features['featureVector'][i] = np.array(feature_list)

        # self.features['featureVector'] = self.numericFeatures['bpmRangeAdjusted'].astype(str) + self.numericFeatures['energyLevel'].astype(str).replace(
        #     {'[': ' ', ']': ' '}) + self.numericFeatures['mood'].astype(str).replace({'[': ' ', ']': ' '}) + self.numericFeatures['subgenre']

    def rankBySimilarity(self, queryFilename):
        self.libraryFeatures = pd.read_csv(self.LibraryFeaturesPath)
        # Initialize df
        self.numericFeatures = pd.DataFrame(columns=['title', 'energyLevel', 'bpmRangeAdjusted', 'mood', 'subgenre'])
        self.numericFeatures['title'] = self.libraryFeatures['title']

        # Data cleaning
        # Fetch each feature and normalize them
        self.getTempoFeature()
        
        self.getEnergyLevels()

        self.getSubGenreTags()

        self.getMoodTags()

        # Concatenate the features
        self.createLibraryFeatures()
        
        queryFeatures = self.features.loc[self.features['title'] == queryFilename]['featureVector'].values[0]

        distances = [self.compute_distance(queryFeatures,recordFeatures) for recordFeatures in self.features['featureVector']]

        smallest_indices = sorted(range(len(distances)), key = lambda sub: distances[sub])[1:self.topK + 1]

        queryResult = self.features['title'][smallest_indices].values

        return queryResult
