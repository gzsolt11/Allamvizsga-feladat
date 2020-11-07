# call python fila as
# python create_baseline_sequence.py processed_data/baseline.csv input_folder length_of_samples

import pandas as pd
import pathlib
import re
import numpy as np
from sklearn import metrics
import os
import shutil
import hashlib
import argparse
from tqdm import tqdm

def create_sequence(input_path,output_path,sample_length):

    # The path to the file
    fn = pathlib.Path(__file__).parent / input_path

    # This list will be printed into the output file
    output = []

    file = open(fn, "r")
    ok = True
    same_user = True
    user = ""
    while True:
        sor = []

        # if there is no next line in the file then break out of the loop
        if not ok:
            break
        
        # Creating a single row from every sample_length rows
        for i in range(0,sample_length):
            if same_user:
                line = file.readline()
                splitted_line = line.split(',')[:4]
            else:
                same_user = True
            if( i == 0 ):
                user = line.split(',')[-1]
            if( i != 0):
                new_user = line.split(',')[-1]
                if(user != new_user):
                    same_user = False
                    break

            sor.append(splitted_line)
            

            if not line:
                ok = False
                break

        # If the user is not the same then we start again from the next user
        if not same_user:
            continue

        if(len(sor) < sample_length):
            break

        # In the chunk list we take the Holdtime1 Holdtime2 PressPress and ReleasePress and concatenate them together
        chunk = []

        H1 = []
        H2 = []
        PP = []
        RP = []
        for i in range(sample_length):
            H1 = H1 + [sor[i][0]]
            H2 = H2 + [sor[i][1]]
            PP = PP + [sor[i][2]]
            RP = RP + [sor[i][3]]
        
        chunk = H1 + H2 + PP + RP + [user.split('\n')[0]]
        
        # Adding the chunk to the output list
        output.append(tuple(chunk))

    file.close()

    # Writing the output list into the output_path        
    write_file = output_path + "/" + "baseline_sequences_" + str(sample_length) + ".csv"
    try:
        os.makedirs(write_file[:-8])
    except:
        pass
    try:
        with open(write_file, "a") as file:
            for entry in output:
                outputstring = ""
                for i in range(sample_length*4+1):
                    if (i < sample_length*4):
                        # Concatenating the elements of H1, H2, PP, RP
                        outputstring = outputstring + str(entry[i]) +","
                    else:
                        # Concatenating the user_id
                        outputstring = outputstring + str(entry[i]) +"\n"
                # Writing the outputstring into the file
                file.write(outputstring)
            file.close()
    except:
        with open(write_file, "w+") as file:
            for entry in output:
                outputstring = ""
                for i in range(sample_length*4+1):
                    if (i < sample_length*4):
                        # Concatenating the elements of H1, H2, PP, RP
                        outputstring = outputstring + str(entry[i]) +","
                    else:
                        # Concatenating the user_id
                        outputstring = outputstring + str(entry[i]) +"\n"
                # Writing the outputstring into the file
                file.write(outputstring)
            file.close()
   
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="input_path", metavar="INPUT_PATH", help="Path to read raw typing data from.")
    parser.add_argument(dest="output_path", metavar="OUTPUT_PATH", help="Path to write processed data to")
    parser.add_argument(dest="sample_length", metavar="SAMPLE_LENGTH", help="Range of the data sequences")

    args = parser.parse_args()

    # Verify that input path exists
    assert os.path.exists(args.input_path), "Specified input path does not exist."

    # Check if path for preprocessed data exists
    if os.path.exists(args.output_path):
        ans = input("All preprocessed data will be overwritten. Do you want to continue? (Y/n) >> ")
        if not(ans == "" or ans.lower() == "y" or ans.lower() == "yes"):
            exit()

    # Verify that sample length exists 
    if not (len(args.sample_length) > 0):
        print("Specified range is not defined.")

    # Verify if the sample length is a number
    try:
        args.sample_length = int(args.sample_length)
    except ValueError:
        print("Specified range is not a number.")
        exit()

    # Creates fresh path for the preprocessed data
    if os.path.exists(args.output_path):
        if "processed_data" not in args.output_path:
            print("Processed data path must include \"processed_data\" as a precaution.")
        else:
            shutil.rmtree(args.output_path)
    os.mkdir(args.output_path)

    # Process the data
    create_sequence(args.input_path,args.output_path,args.sample_length)

    print("Data was preprocessed successfully.")


if __name__ == "__main__":
    main()