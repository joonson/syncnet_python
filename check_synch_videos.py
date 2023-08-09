#!/usr/bin/python
#-*- coding: utf-8 -*-
# Given a folder with input videos, 
# it checks each video to see if the audio is in sync with the video.
# The results are saved in a CSV file, which can be used for filtering or
# for correcting the audio sync.
import subprocess
import os
import re
import time, pdb, argparse, subprocess
import pandas as pd 
from SyncNetInstance import *

# ==================== RESULT CSV FILE ====================
def initialise_results_csv(csv_file_path):
    csv_file = pd.DataFrame(columns=["video_file", "av_offset", "min_dist", "confidence"])
    if os.path.isfile(csv_file_path):
        csv_file = pd.read_csv(csv_file_path)
    return csv_file

def checkAudioVideoSync(file, csv_file):
    reference = file.replace('-', '')
    file = file.replace('-', '\-')
    try:
        # run the command and capture the output
        cmd = "python3 run_pipeline.py --videofile '" + os.path.join(root, file) + f"' --reference '{reference}'"
        print("Calling: " + cmd)
        result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # now execute run_syncnet.py --videofile <video_file> --reference <video_file>
        cmd = "python3 run_syncnet.py --videofile " + os.path.join(root, file) + f" --reference '{reference}'"
        print("Calling: " + cmd)
        result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # run some code if the command succeeds
        print(f"Successfully processed {file}")
        output = result.stdout.decode('utf-8')
        print(output)
        # parse the output
        if "AV offset" not in output:
            print("Error: AV offset not found in output")
            # save the video file to pandas df and set error column to true
            new_row = pd.DataFrame({"video_file": file, "av_offset": "error", "min_dist": "error", "confidence": "error"}, index=[0])
            csv_file = pd.concat([csv_file, new_row], ignore_index=True)
            # save csv_file to disk
            csv_file.to_csv(results_csv_file_path, index=False)
            return 
        av_offset = re.search(r'AV offset:\s+(-?\d+)', output).group(1)
        min_dist = re.search(r'Min dist:\s+([\d.]+)', output).group(1)
        confidence = re.search(r'Confidence:\s+([\d.]+)', output).group(1)
        new_row = pd.DataFrame({"video_file": file, "av_offset": av_offset, "min_dist": min_dist, "confidence": confidence}, index=[0])
        csv_file = pd.concat([csv_file, new_row], ignore_index=True)
        csv_file.to_csv(results_csv_file_path, index=False)

    except subprocess.CalledProcessError as e:
        # run some code if the command fails
        print(f"Error processing {file}: {e.stderr.decode('utf-8')}")
        # ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "SyncNet");

    parser.add_argument('--folder', type=str, default="../data/input/avspeech/train/", help='')
    parser.add_argument('--results', type=str, default="../data/preprocessing/avspeechdataset_results.csv", help='');

    opt = parser.parse_args();
    results_csv_file_path = opt.results
    # create panda dataframe to store results
    # read file from disk if there is one
    csv_file = initialise_results_csv(results_csv_file_path)


    # iterate over all the videos in folder



    for root, dirs, files in os.walk(opt.folder):
        for file in files:
            if file.endswith(".mp4"):
                # check if video is already in CSV file
                
                file_escaped = file.replace('-', '\-')
                file_exists=csv_file["video_file"].str.contains(file_escaped).any()
                print("File exists=" + str(file_exists))
                
                
                if not csv_file["video_file"].str.contains(file_escaped).any():
                    
                    # run pipeline on video
                    checkAudioVideoSync(file, csv_file)
                    


