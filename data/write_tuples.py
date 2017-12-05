#!/usr/bin/env python

import numpy as np
import pickle
import argparse

usage_help = """./write_tuples.py Training_Shuffled_Dataset.txt"""
parser = argparse.ArgumentParser(description=usage_help)
parser.add_argument('txtfile')
args = parser.parse_args()

def write_tuples(filename_data, filename_output):
    with open(filename_data, 'r') as data:
        with open(filename_output, 'w') as output:
            for line_n, line in enumerate(data, start=1):
                fragments = line.split('\t')
                if len(fragments) != 3:
                    print("Line %d contains %d fragments; skipping" %
                            (line_n, len(fragments)))
                    continue
                output.write(fragments[0] + '\n' + fragments[1] + '\n')
                output.write(fragments[1] + '\n' + fragments[2])

filename_data = args.txtfile
filename_output = args.txtfile.replace(".txt", "_Tuples.txt")
write_tuples(filename_data, filename_output)
