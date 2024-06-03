
import xml.etree.cElementTree as et
import pandas as pd
import glob
from os import walk
from os.path import join, basename

    
def generate_labels(input_file, out_dir, tier_name):
    parsed_xml = et.parse(input_file)
    d = parsed_xml.getroot()
    time_dict = {"TIME_SLOT_ID":[], "TIME_VALUE":[]}
    for child in d.find('.//TIME_ORDER'):

        time_dict["TIME_SLOT_ID"].append(child.attrib['TIME_SLOT_ID'])
        time_dict["TIME_VALUE"].append(child.attrib['TIME_VALUE'])

    timeslot_map = pd.DataFrame.from_dict(time_dict)

    anno_dict = {"ANNOTATION_ID":[],"START_TIME":[], "END_TIME":[], "DURATION":[], "ANNOTATION_VALUE":[]}
    tier = d.find(".//TIER[@TIER_ID='" + tier_name + "']")
    if tier is not None:
        for annot in tier.iter('ALIGNABLE_ANNOTATION'):
            anno_dict["ANNOTATION_ID"].append(annot.attrib['ANNOTATION_ID'])
            start = int(timeslot_map[timeslot_map.TIME_SLOT_ID==annot.attrib['TIME_SLOT_REF1']].TIME_VALUE.item())/1000
            end = int(timeslot_map[timeslot_map.TIME_SLOT_ID==annot.attrib['TIME_SLOT_REF2']].TIME_VALUE.item())/1000
            anno_dict["START_TIME"].append(start)
            anno_dict["END_TIME"].append(end)
            anno_dict["DURATION"].append(end - start)
            anno_dict["ANNOTATION_VALUE"].append(annot.find('ANNOTATION_VALUE').text)

    anno = pd.DataFrame.from_dict(anno_dict)
    anno.to_csv(join(out_dir,basename(input_file).split('.')[0]+'.csv'), sep=',',index=False, header=True)

folder_path = '../data/raw/videos_signals/*.eaf'
out_path = '../data/annotation/video_signals/'
files_path = sorted(glob.glob(folder_path))
tier_name = 'action'
for i_file in files_path:
    generate_labels(i_file, out_path, tier_name)

