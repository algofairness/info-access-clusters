import main_pipelines as mp
import ctypes
from pathlib import Path
import subprocess
import sys
import os
import argparse
import configparser
import json


def main():

    parser = argparse.ArgumentParser(description='get path to the config file.')
    parser.add_argument('filepath', help='path to the config file, including filename')
    args = parser.parse_args()
    configFile=args.filepath
    #configFile='config.ini'
    #ConfigParser support https://docs.python.org/3/library/configparser.html#supported-ini-file-structure
    config = configparser.ConfigParser()
    config.read(configFile)
    srcData = config['GENERAL']['srcEdgeListFile']
    dstVectorFile = config['GENERAL']['dstVectorDir']+"/vectors_"+config['GENERAL']['experimentName']+".txt"
    alpha = config['GENERAL']['alphaValue']
    repNumber = config['GENERAL']['repititions']
    simSeeds = config['GENERAL']['simAllSeeds']

    subprocess.Popen(["./C++ code/main", srcData, dstVectorFile, alpha, repNumber, simSeeds]).wait() #run C++ code

    #mp.pipeline_after_vectors()


if __name__=="__main__":
    main()
