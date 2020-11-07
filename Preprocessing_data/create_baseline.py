# Call python file as
# python create_baseline.py input processed_data

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import re
import numpy as np
from sklearn import metrics
import os
import shutil
import hashlib
import argparse
from tqdm import tqdm

KEYS = ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "A",
        "S", "D", "F", "G", "H", "J", "K", "L", "Z", "X", "C",
        "V", "B", "N", "M", "Space", "LShiftKey", "RShiftKey",
        "Back", "Oemcomma", "OemPeriod", "NumPad0", "NumPad1",
        "NumPad2", "NumPad3", "NumPad4", "NumPad5", "NumPad6",
        "NumPad7", "NumPad8", "NumPad9", "D0", "D1", "D2", "D3",
        "D4", "D5", "D6", "D7", "D8", "D9"]

# Adding to files every txt file in the baseline folders inside the input file
def beolvas(read_path):
    files = []
    for i in range(0,3):
        folder = 's'+str(i)
        fn = pathlib.Path(pathlib.Path(read_path) / folder / 'baseline').rglob('*.txt')
        files = files + [x for x in fn]
    return files
    

def parse_data(read_path, output_path):
    files = beolvas(read_path)

    output = []
    
    for file in files:
        # Getting the user_id, session_id, keyboard_id, task_id from the name of the file
        useful_data = str(file).split("\\")[-1].split(".")[0]
        user_id = useful_data[0:3]
        session_id = useful_data[3:4]
        keyboard_id = useful_data[4:5]
        task_id = useful_data[-1]

        # Opening file
        with open(file, "r") as file0:

            # In this list we store every key with the press or release times
            key_with_times = []

            for line in file0:
                # splitting the lines
                key, action, time = line.split()

                # If the key is not in the list
                if key not in KEYS:
                    continue

                # If the action is keyDown we add it to the list
                if action == "KeyDown":
                    key_with_times.append([key, time, None])

                # If the action is KeyUp we add the release time next to the press time
                elif action == "KeyUp":
                    for pressedKey in key_with_times[::-1]:
                        if pressedKey[0] == key:
                            if pressedKey[2] is None:
                                pressedKey[2] = time
                            else:
                                break

            
            for i in range(len(key_with_times)):
                if i == len(key_with_times) - 1:
                    break
                try:
                    # Calculating the Holdtime1, holdtime2, PressPress, ReleasePress times from the rows
                    H1 = int(key_with_times[i][2]) - int(key_with_times[i][1])
                    H2 = int(key_with_times[i+1][2]) - int(key_with_times[i+1][1])
                    PP = int(key_with_times[i+1][1]) - int(key_with_times[i][1])
                    RP = int(key_with_times[i+1][1]) - int(key_with_times[i][2])
                    # If the PP and RP looks accurate we add it to the output
                    if PP < 1000 and abs(RP) < 1000:
                        output.append((H1,H2, PP, RP, session_id, keyboard_id, task_id, user_id))
                except:
                    pass
        
        
    # Write processed data to file
    write_file = output_path + "/" + "baseline.csv"
    try:
        os.makedirs(write_file[:-8])
    except:
        pass
    try:
        with open(write_file, "a") as file:
            for entry in output:
                file.write(str(entry[0]) + "," + str(entry[1]) + "," + str(entry[2]) + "," + str(entry[3]) + "," + str(entry[4]) + "," + str(entry[5]) +"," + str(entry[6]) + "," + str(entry[7]) + "\n")
            file.close()
    except:
        with open(write_file, "w+") as file:
            for entry in output:
                file.write(str(entry[0]) + "," + str(entry[1]) + "," + str(entry[2]) + "," + str(entry[3]) + "," + str(entry[4]) + "," + str(entry[5]) +"," + str(entry[6]) + "," + str(entry[7]) + "\n")
            file.close()
        
       
    
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="input_path", metavar="INPUT_PATH", help="Path to read raw typing data from.")
    parser.add_argument(dest="output_path", metavar="OUTPUT_PATH", help="Path to write processed data to")

    args = parser.parse_args()

    # Verify that input path exists
    assert os.path.exists(args.input_path), "Specified input path does not exist."

    # Check if path for preprocessed data exists
    if os.path.exists(args.output_path):
        ans = input("All preprocessed data will be overwritten. Do you want to continue? (Y/n) >> ")
        if not(ans == "" or ans.lower() == "y" or ans.lower() == "yes"):
            exit()

    # Creates fresh path for the preprocessed data
    if os.path.exists(args.output_path):
        if "processed_data" not in args.output_path:
            print("Processed data path must include \"processed_data\" as a precaution.")
        else:
            shutil.rmtree(args.output_path)
    os.mkdir(args.output_path)

    # Process the data
    parse_data(args.input_path, args.output_path)

    print("Data was preprocessed successfully.")


if __name__ == "__main__":
    main()

