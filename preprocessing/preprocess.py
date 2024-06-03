import xml.etree.cElementTree as et
import pandas as pd
import glob
from os import walk
from os.path import join, basename
import re
import numpy as np
import csv
from scipy.io import arff
import math 
from collections import Counter
from scipy import interpolate
import statistics
from datetime import datetime
#from numpy_ext import rolling_apply

ebidir = './Akram_EBI/'
annodir  = './Session_Videos/*.eaf'

# Parameters
ebi_sampling_rate = 390  # in Hz


def read_anno(input_file):
    parsed_xml = et.parse(input_file)
    d = parsed_xml.getroot()
    time_dict = {"TIME_SLOT_ID":[], "TIME_VALUE":[]}
    for child in d.find('.//TIME_ORDER'):

        time_dict["TIME_SLOT_ID"].append(child.attrib['TIME_SLOT_ID'])
        time_dict["TIME_VALUE"].append(child.attrib['TIME_VALUE'])

    timeslot_map = pd.DataFrame.from_dict(time_dict)
    timeslot_map

    anno_dict = {"ANNOTATION_ID":[],"START_TIME":[], "END_TIME":[], "DURATION":[], "ANNOTATION_VALUE":[]}
    for annot in d.iter('ALIGNABLE_ANNOTATION'):
        anno_dict["ANNOTATION_ID"].append(annot.attrib['ANNOTATION_ID'])
        start = timeslot_map[timeslot_map.TIME_SLOT_ID==annot.attrib['TIME_SLOT_REF1']].TIME_VALUE.item()
        end = timeslot_map[timeslot_map.TIME_SLOT_ID==annot.attrib['TIME_SLOT_REF2']].TIME_VALUE.item()
        anno_dict["START_TIME"].append(start)
        anno_dict["END_TIME"].append(end)
        anno_dict["DURATION"].append(int(end) - int(start))
        anno_dict["ANNOTATION_VALUE"].append(annot.find('ANNOTATION_VALUE').text)

    anno = pd.DataFrame.from_dict(anno_dict)
    return anno


def process(filePath, duration, sampling_rate):

    # Calculate total number of samples
    total_samples = int(duration * sampling_rate)

    # Read CSV file
    df = pd.read_csv('your_file.csv')

    # Extract the 'Signal' column
    signal_data = df['Signal']

    # Extract data based on the total number of samples
    extracted_data = signal_data[:total_samples]


def main():
    S_number = '01'
    S_Age = 'M'
    filePath = '/Users/szi02/Documents/PRE-Fall/data/pre_processed/EBI/10'+ f'{S_number:02d}' + '_'+  S_Age + '/'  
    run = '10'+ f'{S_number:02d}' + '_'+  S_Age
    join(filePath,basename(run).split('.')[0]+'.csv') 

    process(filePath, duration, sampling_rate)