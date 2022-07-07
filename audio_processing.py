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

cwheel = camelotWheel.CamelotWheel()

def estimate_bpm(file,verbose=False):

    rhy = ess.RhythmExtractor2013()
    loader = ess.MonoLoader(filename=file)
    loader.audio >> rhy.signal
    pool = essentia.Pool()

    rhy.ticks >> None
    rhy.confidence >> None
    rhy.estimates >> None
    rhy.bpmIntervals >> None

    rhy.bpm >> (pool, 'bpm')
    essentia.run(loader)
    bpm = pool['bpm']
    if verbose:
        print('Essentia BPM: ')
        print(bpm)

    return bpm

def estimate_key(file,verbose=False):

    loader = ess.MonoLoader(filename=file)
    framecutter = ess.FrameCutter(frameSize=4096, hopSize=2048, silentFrames='noise')
    windowing = ess.Windowing(type='blackmanharris62')
    spectrum = ess.Spectrum()
    spectralpeaks = ess.SpectralPeaks(orderBy='magnitude',
                                      magnitudeThreshold=0.00001,
                                      minFrequency=20,
                                      maxFrequency=3500,
                                      maxPeaks=60)

    # Use default HPCP parameters for plots, however we will need higher resolution
    # and custom parameters for better Key estimation

    hpcp = ess.HPCP()
    hpcp_key = ess.HPCP(size=36, # we will need higher resolution for Key estimation
                        referenceFrequency=440, # assume tuning frequency is 44100.
                        bandPreset=False,
                        minFrequency=20,
                        maxFrequency=3500,
                        weightType='cosine',
                        nonLinear=False,
                        windowSize=1.)

    key = ess.Key(profileType='edma', # Use profile for electronic music
                  numHarmonics=4,
                  pcpSize=36,
                  slope=0.6,
                  usePolyphony=True,
                  useThreeChords=True)

    # Use pool to store data
    pool = essentia.Pool()

    # Connect streaming algorithms
    loader.audio >> framecutter.signal
    framecutter.frame >> windowing.frame >> spectrum.frame
    spectrum.spectrum >> spectralpeaks.spectrum
    spectralpeaks.magnitudes >> hpcp.magnitudes
    spectralpeaks.frequencies >> hpcp.frequencies
    spectralpeaks.magnitudes >> hpcp_key.magnitudes
    spectralpeaks.frequencies >> hpcp_key.frequencies
    hpcp_key.hpcp >> key.pcp
    hpcp.hpcp >> (pool, 'tonal.hpcp')
    key.key >> (pool, 'tonal.key_key')
    key.scale >> (pool, 'tonal.key_scale')
    key.strength >> (pool, 'tonal.key_strength')

    essentia.run(loader)
    if verbose:
        print((pool['tonal.key_key'], pool['tonal.key_scale']))

    return  pool['tonal.key_key'], pool['tonal.key_scale']

def analyzeAudioLibrary(libraryPath):
    descriptors = []
    tracks = os.listdir(libraryPath)
    tracksfilesPath = [os.path.join(libraryPath,track) for track in tracks]
    for rec_file in tracksfilesPath:
        print(rec_file)
        rec_bpm = estimate_bpm(rec_file)
        rec_key = estimate_key(rec_file)
        rec_key_numeric = cwheel.essentiaToCamelotNotation[" ".join(rec_key)]
        descriptors.append([os.path.basename(rec_file),rec_key_numeric,rec_bpm])

    return pd.DataFrame(descriptors,columns=['Record Name','KEY','BPM'])

def searchSimilarRecords(record_file,library_descriptors,criteria):
    # Retrieve if songs have similar keys and those within 3% bpm offset from the query, a maximum of k records
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
def nearestRecordsByBPM(query_bpm,library_descriptors):

    K=0.03 # % Desviation from query track in BPM
    database_bpm = list(library_descriptors['BPM'].values)
    idxs = withinKperCent(query_bpm,database_bpm,K)
    records = list(library_descriptors['Record Name'].values)

    if len(idxs) == 0:
        print("Not enough records")
        return None
    else:
        nearestRecords = [records[i] for i in idxs]
        return nearestRecords

def nearestRecordsByKey(query_key,library_descriptors):

    database_key = list(library_descriptors['KEY'].values)
    idxs = findRecordsinKey(query_key,database_key)
    records = list(library_descriptors['Record Name'].values)
    return [records[i] for i in idxs]

def findRecordsinKey(query,database):

    similar_records_idxs = []
    for i,db_key in enumerate(database):
        print(query,db_key)
        if cwheel.similar_keys(query,db_key):
            similar_records_idxs.append(i)
    if len(similar_records_idxs) == 0:
        print("No enough records in key")
        similar_records_idxs = None
    return similar_records_idxs

def withinKperCent(query_bpm,database_bpms,K):

    values = np.asarray(database_bpms)
    deltas = np.abs(values - query_bpm)/query_bpm
    return np.where(deltas < K)[0].tolist()


def handle_uploaded_audio_file(uploaded_file):
    a = pydub.AudioSegment.from_file(file=uploaded_file, format=uploaded_file.name.split(".")[-1])

    channel_sounds = a.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]

    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples[0].typecode).max

    return fp_arr[:, 0], a.frame_rate


def plot_wave(y, sr):
    fig, ax = plt.subplots()
    n_sec = 5
    img = librosa.display.waveplot(y[:n_sec*sr], sr=sr, x_axis='time', ax=ax)

    return plt.gcf()


def plot_transformation(y, sr, transformation_name):
    D = librosa.stft(y)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
    # ax.set(title=transformation_name)
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    return plt.gcf()
def plot_audio_transformations(y, sr, pipeline: audiomentations.Compose):
    cols = [1, 1, 1]

    col1, col2, col3 = st.columns(cols)
    with col1:
        st.markdown(f"<h4 style='text-align: center; color: black;'>Original</h5>",
                    unsafe_allow_html=True)
        st.pyplot(plot_transformation(y, sr, "Original"))
    with col2:
        st.markdown(f"<h4 style='text-align: center; color: black;'>Wave plot </h5>",
                    unsafe_allow_html=True)
        st.pyplot(plot_wave(y, sr))
    with col3:
        st.markdown(f"<h4 style='text-align: center; color: black;'>Audio</h5>",
                    unsafe_allow_html=True)
        spacing()
        st.audio(create_audio_player(y, sr))
    st.markdown("---")

    y = y
    sr = sr
    for col_index, individual_transformation in enumerate(pipeline.transforms):
        transformation_name = str(type(individual_transformation)).split("'")[1].split(".")[-1]
        modified = individual_transformation(y, sr)
        fig = plot_transformation(modified, sr, transformation_name=transformation_name)
        y = modified

        col1, col2, col3 = st.columns(cols)

        with col1:
            st.markdown(f"<h4 style='text-align: center; color: black;'>{transformation_name}</h5>",
                        unsafe_allow_html=True)
            st.pyplot(fig)
        with col2:
            st.markdown(f"<h4 style='text-align: center; color: black;'>Wave plot </h5>",
                        unsafe_allow_html=True)
            st.pyplot(plot_wave(modified, sr))
            spacing()

        with col3:
            st.markdown(f"<h4 style='text-align: center; color: black;'>Audio</h5>",
                        unsafe_allow_html=True)
            spacing()
            st.audio(create_audio_player(modified, sr))
        st.markdown("---")
        plt.close("all")
