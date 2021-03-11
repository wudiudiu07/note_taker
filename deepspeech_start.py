
import deepspeech
import wave
import re
import numpy as np
import sys
import glob
import os
from ffmpy import FFmpeg
import ntpath
import scipy.io.wavfile
import scipy.signal
import shutil
import argparse
import subprocess
from tqdm import tqdm
from audioProcessing import extract_audio, convert_samplerate
import segmentAudio

def convert_video(filepath):
	base = ntpath.basename(filepath)
	filename = os.path.splitext(base)[0]
	new_name = filename + '.wav'
	ff = FFmpeg(
		executable = './ffmpeg/bin/ffmpeg.exe',
		inputs={filepath: None},
		outputs={new_name: None}
		)
	ff.run()
	to_deepspeech(new_name)

def sort_alphanumeric(data):
    """Sort function to sort os.listdir() alphanumerically
    Helps to process audio files sequentially after splitting 
    Args:
        data : file name
    """
    
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    
    return sorted(data, key = alphanum_key)

def ds_process_audio(ds,filepath,file_handle):
	w = wave.open(filepath, 'r')
	frames = w.getnframes()
	fs = w.getframerate()#sampling rate
	buffer = w.readframes(frames)
	desired_sample_rate = ds.sampleRate()
	if fs == desired_sample_rate:
		data16 = np.frombuffer(buffer, dtype=np.int16)
	else:
		data16 = convert_samplerate(filepath,desired_sample_rate)
	w.close()
	
	text = ds.stt(data16)#use the deepspeech model to perform speech-to-text
	file_handle.write(text+"\n")
	
	

def to_deepspeech(filepath):
	
	model_file_path = 'deepspeech-0.9.3-models.pbmm'
	scorer_path = 'deepspeech-0.9.3-models.scorer'
	ds = deepspeech.Model(model_file_path)
	ds.enableExternalScorer(scorer_path)
	
	base_directory = os.getcwd() #return current directory
	output_directory = os.path.join(base_directory,"output")
	audio_directory = os.path.join(base_directory,"temp")#save temp audio segment
	#output text file
	audio_file_name = filepath.split(os.sep)[-1].split(".")[0]
	filename = ntpath.basename(filepath).split(".")[0]
	txt_file_name = os.path.join(output_directory,filename+".txt")
	file_handle = open(txt_file_name,"w+")
	
	#split silent parts in audio file
	segmentAudio.silenceRemoval(filepath,audio_directory)  
	
	print("Running interface:")
	for file in tqdm(sort_alphanumeric(os.listdir(audio_directory))):
		audio_segment_path = os.path.join(audio_directory,file)
		ds_process_audio(ds, audio_segment_path,file_handle)
	
	print("finished! output is",txt_file_name)
	file_handle.close()
	
	##clean directory and temp file
	#shutil.rmtree(audio_directory)
	#os.mkdir(audio_directory)
	
def main():
    filepath = sys.argv[3]#path, get rid of 'File Path'
	
    name,extension = os.path.splitext(filepath)

    #video to audio file
    if (extension == ".wav"):
        print("Input is wave file")
        #if wav file, detect downsample or upsample
        #fs = wave.open(filepath,'rb').getframerate()
        to_deepspeech(filepath)
    elif (extension == ".mp4"):
        print("input is mp4 file")
        #convert 
        convert_video(filepath)


if __name__ == "__main__": 
 # if you call this script from the command line (the shell) it will
 # run the 'main' function
	main()