import tkinter as tk
from tkinter import filedialog
from tkinter import *
import os
import wave
from pathlib import Path

if __name__ == "__main__":
 # if you call this script from the command line (the shell) it will
 # run the 'main' function
	P_window = tk.Tk()
	P_window.geometry("500x250") 
	P_window.title("Automated Note Taker")
	P_window.config(background = "white")
	label_file_name = tk.Label(P_window,text = "File path:     ", width = 100, height = 4, fg = "blue")
	filename = label_file_name.cget("text")
	button_explore = tk.Button(P_window, text = "Browse Files", command = lambda:browseFiles(filename)).pack() #browser
	label_file_name.pack()
	
	def browseFiles(filename):
		filename = filedialog.askopenfilename()##open a file chooser
		label_file_name.configure(text="File Path: "+filename)
		#label_file_name.configure(filename)
	
	def call_deepspeech(filename):
	#file_path = 'C:/Users/wuzis/Documents/deepspeech/audio/mix_.wav'
		file_path = label_file_name.cget("text")
		os.system('python deepspeech_start.py'+' '+ file_path)
	
	# checkbox for mp4 or wav
	mp4 = IntVar()
	chk_1 = Checkbutton(P_window, text = "Video",variable = mp4).pack()
	wav = IntVar()
	chk_2 = Checkbutton(P_window, text = "Audio",variable = wav).pack()
	spe_txt = tk.Button(P_window, text = "start speech to text", command =lambda:call_deepspeech(filename)).pack()#start deepspeech
	#spe_txt = tk.Button(P_window, text = "start speech to text", command =call_deepspeech).pack()#start deepspeech
	
	tk.Button(P_window,text='Exit',command=P_window.quit).pack()
	P_window.mainloop()
	#filename = 'C:/Users/wuzis/Documents/deepspeech/audio/mix_.wav'
	#os.system('python deepspeech_start.py'+' '+ filename)	


