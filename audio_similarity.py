import os
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
from sklearn.preprocessing import StandardScaler


class SimilarityEngine:

    def __init__(self,queryFeaturesPath,libraryFeaturesPath):
        self.queryFeaturesPath = queryFeaturesPath
        self.LibraryFeaturesPath = libraryFeaturesPath
        self.libraryFeatures = None
        self.topK = 5

    def set_queryPath(queryFeaturesPath):
        """
         Every time the user uploads a new query track
        """
        self.queryFeaturesPath = queryFeaturesPath

    def set_libraryPath(queryFeaturesPath):
        """
         Every time the user uploads a new analyzed library of tracks
        """
        self.libraryFeaturesPath = libraryFeaturesPath

    def normalizeSubGenreTags(subGenreTags):
        "Transform to probabilities the vector of subGenreTags"
        normalizer = 1 / float( sum(subGenreTags) )
        normalizedsubGenreTags = [x * normalizer for x in subGenreTags]
        return normalizedsubGenreTags

    def mapEnergyLevels(self,energyLevels):
        classesToLabels = ['low':1,'high':3,'medium':2,'variable':2]
        # TODO
        return energyLevels

    def getTempoFeature(self,track):
        """
        Get tempo from the library features for the particular track. Single float value.
        """
        features_df = pd.read_csv(self.libraryFeaturesPath)
        # TODO
        return BPM

    def getCyaniteFeatures(self,track):
        """ Return weights of  mood vector and the filtered subgenre vector, in this order, for a given query
        1) Map energy energyLevels
        2) Normalize to 1 the subGenreTags
        Return list of floats
        """
        features_df = pd.read_csv(self.libraryFeaturesPath)
        # TODO
        return energyLevels,moodTags subGenreTags

    def normalizeFeatures(self,features, libraryFeatures):
        """
        Brings all features to same scale. Mean removal for the moment
        """
        scaler = StandardScaler()
        scaler.fit(self.libraryFeatures)
        normalizedFeatures = scaler.transform(features)
        return normalizedFeatures

    def compute_similarity(self,queryFeatures,recordFeatures):

        """ Two numpy arrays: queryFeatures and libraryFeatures
        Returns similarity value
        """
        queryFeatures = np.array(queryFeatures)
        recordFeatures = np.array(recordFeatures)
        normQueryFeatures = self.normalizeFeatures(queryFeatures)
        normRecordFeatures = self.normalizeFeatures(recordFeatures)
        similarity = scipy.spatial.distance.cosine(normQueryFeatures,normRecordFeatures)
        return similarity

    def createLibraryFeatures(self):
        """
        Schema for the features : [Tempo,energyLevels(4),moodVector(X),subGenreTags(Y)]
        """
        features_df = pd.read_csv(self.libraryFeaturesPath)
        libraryFeatures = []
        for recordName in features_df['Record Name']:
            libraryFeatures.append([self.getTempoFeature(recordName)].extend(self.getCyaniteFeatures(recordName)))
        return libraryFeatures

    def rankBySimilarity(self,queryFilename):
        queryFeatures = []
        queryFeatures.append(self.getTempoFeature(queryFilename))
        queryFeatures.append(self.getCyaniteFeatures(queryFilename))
        self.libraryFeatures = self.createLibraryFeatures()
        queryResults = [self.compute_similarity(queryFeatures,recordFeatures) for recordFeatures in libraryFeatures]
        # Sort results, keep the original indexes and map to record names!!
        # TODO
        return recordRecommendations[:self.topK]
