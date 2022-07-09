import io
import librosa
import numpy as np
import streamlit as st
import audiomentations
from matplotlib import pyplot as plt
import librosa.display
from scipy.io import wavfile
import wave
import pydub
from audio_processing import *
import pandas as pd

plt.rcParams["figure.figsize"] = (10, 7)
plt.rcParams['agg.path.chunksize'] = 10000


def create_audio_player(audio_data, sample_rate):
    virtualfile = io.BytesIO()
    wavfile.write(virtualfile, rate=sample_rate, data=audio_data)

    return virtualfile

def spacing():
    st.markdown("<br></br>", unsafe_allow_html=True)


def action(file_uploader):
    if file_uploader is not None:
        y, sr = handle_uploaded_audio_file(file_uploader)
    else:
        st.error('Not selected file')
    fig = plot_wave(y, sr)
    st.pyplot(fig)

def fileSelector(folder_path):
    filenames = os.listdir(folder_path)
    print(filenames)
    selected_filename = st.sidebar.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


def main():
    placeholder = st.empty()
    placeholder2 = st.empty()
    placeholder.markdown("# Nextrack Finder\n"
                         "### Upload a track library ( or pre-processed library) and query track in  the sidebar and start mixing!\n"
                         "Then click \"Get recommended tracks\" to start!\n ")

    #st.sidebar.markdown("Upload an audio library and analyze:")
    with st.sidebar.form(key='upload_button'):
       libraryFilePath = st.text_input(label = "Please enter the input path: " )
       analyze_button = st.form_submit_button(label='Load and analyze an audio library')
       if analyze_button:
           try:
               with st.spinner(text="Analyzing..."):
                   descriptors_df = analyzeAudioLibrary(libraryFilePath)
               output_path = "featuresDB/KeyTempo_features.csv"
               st.session_state['libraryPath'] = libraryFilePath
               st.session_state['descriptorsPath'] = output_path
               descriptors_df.to_csv (output_path, index = False, header=True)
               st.success("Succesfully analyzed library")
           except FileNotFoundError:
               st.warning("Input a valid library path")
    #st.sidebar.markdown("Or, load your database of features for an analyzed library")
    with st.sidebar.form(key='upload_button_features'):
            featuresFilePath = st.text_input(label = "Please enter the input path to the database features: " )
            load_features_button = st.form_submit_button(label='Load features (*.csv)')
            if load_features_button:
                if os.path.isfile(featuresFilePath):
                    st.session_state['descriptorsPath'] = featuresFilePath
                    st.success("Succesfully loaded features")
                else:
                    st.warning("Input a valid database path")

    st.sidebar.markdown("---")
    st.sidebar.markdown("Select a query track")
    #placeholder.empty()
    #placeholder2.empty()
    uploaded_file = st.sidebar.file_uploader("Upload a query track")

    if uploaded_file:
        #st.write("Filename: ", uploaded_file.name)
        if 'libraryPath' in st.session_state or 'descriptorsPath' in st.session_state :
        #with st.sidebar.form(key='upload_query'):
        #try:
                queryFilename = uploaded_file.name
                st.session_state["queryFilename"] = queryFilename
                st.write('You selected `%s`' %st.session_state["queryFilename"] )
        #except FileNotFoundError:
        #        st.warning("Please input a valid query path")
        else:
            st.error("Specify a library path above")

    if st.button("Get recommended tracks: "):
        # Call the SimilarityEngine class and update this part.
        # TODO
        placeholder.empty()
        placeholder2.empty()
        if "descriptorsPath" in st.session_state:
            descriptors_df = pd.read_csv(st.session_state['descriptorsPath'])
            similarTracks = searchSimilarRecords(st.session_state["queryFilename"],descriptors_df,'harmonic')
            st.title("Similar tracks")
            st.write(similarTracks)
        else:
            st.write("No library is analyzed")


if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="Nextrack Finder")
    main()
