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


class SimilarityEngine:

    def __init__(self,queryFeaturesPath,databaseFeaturesPath,):
        self.queryFeaturesPath = queryFeaturesPath
        self.libraryFeaturesPath = databaseFeaturesPath

    def normalizeSubGenreTags():
    def tra
    def loadCyaniteFeatures(self):
        "Return Mood vector, energy level"
        self.features_df = pd.read_csv(self.queryFeaturesPath)
        ## TODO
        return None
    def compute_similarity()
    def rankBySimilarity(self):
        self
        result = [self.compute_similarity(queryFeatures,databaseFeatures) for x in self.df['col']]
        result.sort...


def searchSimilarRecords(record_file,library_descriptors,criteria):
    # Retrieve if songs have similar keys and those within 3% bpm offset from the query
    query_name = os.path.basename(record_file)
    if query_name in library_descriptors["Record Name"].values:
        query_bpm = library_descriptors[library_descriptors["Record Name"] == query_name]['BPM'].item()
        query_key_numeric = library_descriptors[library_descriptors["Record Name"] == query_name]['KEY'].item()
    else:
        print(" Record not in library. Analyzing record ...")
        query_bpm = estimate_bpm(record_file)
        query_key = estimate_key(record_file)
        query_key_numeric = cwheel.essentiaToCamelotNotation[" ".join(query_key)]

    records_by_key = set(nearestRecordsByKey(query_key_numeric,library_descriptors))
    records_by_BPM = nearestRecordsByBPM(query_bpm,library_descriptors)
    intersection = records_by_key.intersection(records_by_BPM)
    if len(list(intersection)) == 0:
        if criteria == "tempo":
            print("Not in key and within BPM range. Returning only BPM similar records")
            return records_by_BPM
        elif criteria == "harmonic":
            print("Not in key and within BPM range. Returning only Key similar records")
            return records_by_key
    else:
        return list(intersection)
