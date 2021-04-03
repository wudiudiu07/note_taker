import deepspeech
import wave
import re
import numpy as np
import sys
import pathlib
import glob
import os
import pydub
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
import truecase
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

def correct_output(in_file,out_file):
    PUNCTUATION_MAPPING = {" !EXCLAMATIONMARK" : "!", " .PERIOD" : ".", " :COLON": ":", " ,COMMA" : ",", " ;SEMICOLON": ";" ," -DASH": "-"," ?QUESTIONMARK" : "?"}
    in_f = open(in_file,'r')
    out_f = open(out_file,'w')
    contents = in_f.read()
    for x in PUNCTUATION_MAPPING:
        contents = contents.replace(x,PUNCTUATION_MAPPING[x])
    contents = truecase.get_true_case(contents)
    out_f.write(contents)
    in_f.close()
    out_f.close()


def convert_video(filepath):
    base = ntpath.basename(filepath)
    filename = os.path.splitext(base)[0]
    new_name = filename + '.wav'
    ff = FFmpeg(
        #executable = './ffmpeg/bin/ffmpeg.exe',
        inputs={filepath: None},
        outputs={new_name: None}
        )
    ff.run()
    to_deepspeech(new_name)

def convert_mp3(filepath):
    base = ntpath.basename(filepath)
    filename = os.path.splitext(base)[0]
    #pydub.AudioSegment.Converter = os.getcwd() + "\ffmpeg\bin\ffmpeg.exe"
    #pydub.AudioSegment.ffprobe = os.getcwd() + "\ffmpeg\bin\ffprobe.exe"
    #ADD THE PATH OF FFMPEG first
    audio = pydub.AudioSegment.from_mp3(filepath)
    new_name = filename + '.wav'
    audio.export(new_name,format="wav")
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
    txt_file_name = os.path.join(output_directory,filename+"_temp1.txt")
    file_handle = open(txt_file_name,"w+")
    
    #split silent parts in audio file
    segmentAudio.silenceRemoval(filepath,audio_directory)  
    
    print("Running interface:")
    for file in tqdm(sort_alphanumeric(os.listdir(audio_directory))):
        audio_segment_path = os.path.join(audio_directory,file)
        ds_process_audio(ds, audio_segment_path,file_handle)
    
    file_handle.close()
    
    print("\nSpeech-to-Text Conversion Complete, restoring Punctuation...")
    
    # Running commands to punctuate deepspeech output
    punct2_script = os.path.join(base_directory,"punctutator2_theano\punctuator2-master\punctuator.py")
    punct2_model = os.path.join(base_directory,"punctutator2_theano\models\INTERSPEECH-T-BRNN.pcl")
    punct_output = os.path.join(output_directory,filename+"_temp2.txt")
    final_output = os.path.join(output_directory,filename+".txt")
    
    subprocess.run(["python2",punct2_script,punct2_model,punct_output,txt_file_name]) #punct_output_txt
    correct_output(punct_output,final_output)
    
    
    print("\nfinished! output is",final_output,"\n") #final_output_txt
    print("Summarization starts...\n")
    
    
    # deleting original, unpuncutatued deepspeech output
    if os.path.exists(txt_file_name):
      os.remove(txt_file_name)
    else:
      print("The file does not exist")
    
    if os.path.exists(punct_output):
      os.remove(punct_output)
    else:
      print("The file does not exist")
        
    
    # Text Summarization
    LANGUAGE = "english"
    f = open(final_output,'r')
    lines = f.read().find(".")
    #print(lines,"\n")
    f.close()
    
    SENTENCES_COUNT = int(lines/4) #Quarter of the content
    # print(SENTENCES_COUNT,"\n")

    parser = PlaintextParser.from_file(final_output, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)

    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    Final = os.path.join(output_directory,filename+"_Final.txt")
    F = open(Final,"w")
    
    print("Your Output will be shown below and saved in the output directory.\n")
    for sentence in summarizer(parser.document, SENTENCES_COUNT):
        print(sentence)
        print(sentence, file=F)
    
    F.close()
    print("\nEverything is done!")
    
    ##clean directory and temp file
    tmp_wav = os.path.join(base_directory,filename + ".wav")
    if os.path.exists(tmp_wav):
        os.remove(tmp_wav)
        
    shutil.rmtree(audio_directory)
    os.mkdir(audio_directory)
    
    
    #if os.path.exists(final_output):
        #os.remove(final_output)
    #else:
        #print("The file does not exist")
        
    #os.rename(tmp_f,os.path.join(output_directory,filename+".txt"))
    

 
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
    elif (extension == ".mp3"):
        print("input is mp3 file")
        convert_mp3(filepath)


if __name__ == "__main__": 
 # if you call this script from the command line (the shell) it will
 # run the 'main' function
    main()