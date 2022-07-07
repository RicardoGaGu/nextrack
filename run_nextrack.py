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
    selected_filename = st.sidebar.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


def main():
    placeholder = st.empty()
    placeholder2 = st.empty()
    placeholder.markdown("# Nextrack Finder\n"
                         "### Upload a track library in  the sidebar and start mixing!\n"
                         "Then click \"Get recommended tracks\" to start!\n ")

    st.sidebar.markdown("Upload an audio library:")
    with st.sidebar.form(key='upload_button'):
       libraryFilePath = st.text_input(label = "Please enter the path to the library" )
       analyze_button = st.form_submit_button(label='Analyze')
       if analyze_button:
           try:
               with st.spinner(text="Analyzing..."):
                   descriptors_df = analyzeAudioLibrary(libraryFilePath)
               output_path = "data/descriptors_library.csv"
               st.session_state['libraryPath'] = libraryFilePath
               st.session_state['descriptorsPath'] = output_path
               descriptors_df.to_csv (output_path, index = False, header=True)
           except FileNotFoundError:
               st.warning("Input a valid library path")



    st.sidebar.markdown("---")
    st.sidebar.markdown("Select a query track")
    #placeholder.empty()
    #placeholder2.empty()
    if 'libraryPath' in st.session_state:
        try:
            queryFilename = fileSelector(st.session_state['libraryPath'])
            st.session_state["queryFilename"] = queryFilename
        except FileNotFoundError:
            st.warning("Please input a valid query path")
    else:
        st.error("Specify a library path above")

    if st.button("Get recommended tracks: "):
        placeholder.empty()
        placeholder2.empty()
        if "descriptorsPath" in st.session_state:
            st.write('You selected `%s`' %st.session_state["queryFilename"] )
            descriptors_df = pd.read_csv(st.session_state['descriptorsPath'])
            similarTracks = searchSimilarRecords(st.session_state["queryFilename"],descriptors_df,'harmonic')
            st.title("Similar tracks")
            st.write(similarTracks)
        else:
            st.write("No library is analyzed")


if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="Nextrack Finder")
    main()
