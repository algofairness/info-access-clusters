import main_pipelines as mp
import ctypes
from pathlib import Path
import subprocess
import sys
import os
import argparse
import configparser
import json
import vector_analysis

def main():

    parser = argparse.ArgumentParser(description='get path to the config file.')
    parser.add_argument('filepath', help='path to the config file, including filename')
    args = parser.parse_args()
    configFile=args.filepath
    #configFile='config.ini'
    #ConfigParser support https://docs.python.org/3/library/configparser.html#supported-ini-file-structure
    config = configparser.ConfigParser()
    config.read(configFile)
    srcEdges = config['GENERAL']['srcEdgeListFile']
    srcNodes = config['GENERAL']['srcNodeListFile']
    dstVectorFile = config['GENERAL']['dstVectorDir']+"/vectors_"+config['GENERAL']['experimentName']+".txt"
    alpha1 = config['GENERAL']['alpha1']
    alpha2 = config['GENERAL']['alpha2']
    repNumber = config['GENERAL']['repititions']
    simSeeds = config['GENERAL']['simAllSeeds']


    list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for a1 in list:
        for a2 in list:
            subprocess.Popen(["./C++ code/main", srcEdges, dstVectorFile, str(a1), str(a2), repNumber, simSeeds, srcNodes]).wait() #run C++ code
            vector_analysis.pearson_analysis(srcNodes, dstVectorFile, a1, a2)


    #subprocess.Popen(["./C++ code/main", srcEdges, dstVectorFile, alpha1, alpha2, repNumber, simSeeds, srcNodes]).wait() #run C++ code
    #vector_analysis.pearson_analysis(srcNodes, dstVectorFile)

if __name__=="__main__":
    main()
