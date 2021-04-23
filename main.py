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
import data_rep

#ConfigParser support https://docs.python.org/3/library/configparser.html#supported-ini-file-structure

def main():

    parser = argparse.ArgumentParser(description='get path to the config file.')
    parser.add_argument('filepath', help='path to the config file, including filename')
    args = parser.parse_args()
    configFile=args.filepath #get configFile
    #configFile="PATH TO CONFIGFILE" #uncomment this line to overwrite configFile
    config = configparser.ConfigParser() #make config object
    config.read(configFile) #read configFile

    #GLOBAL CONFIG VARIABLES (ENSURE THIS MATCHES CONFIGFILE NAMES/FORMAT)
    #[GENERAL]
    experimentName = config['GENERAL']['experimentName']
    generateVectors = config['GENERAL']['generateVectors']
    simAllSeeds = config['GENERAL']['simAllSeeds']
    repititions = config['GENERAL']['repititions']
    alphaListStr = config['GENERAL']['alphaList'] #must be changed to string of floats

    #[FILES]
    inEdgesFile = config['FILES']['inEdgesFile']
    inNodesFile = config['FILES']['inNodesFile']
    outVectorDir = config['FILES']['outVectorDir']
    inVectorDir = config['FILES']['inVectorDir']
    outAnalysisDir = config['FILES']['outAnalysisDir']

    #GLOBAL PYTHON VARIABLES (generated from config variables)
    alphaListFlt = [float(item) for item in alphaListStr.split(',')] #usable string of floats
    outAnalysisFile = outAnalysisDir + experimentName + "Analysis" + ".txt"
    outHeatMapFile = outAnalysisDir + experimentName + "Heatmap" + ".txt"

    with open(outAnalysisFile, 'a') as f:
        out = "EXPERIMENT: " + experimentName + "(nodeList: " + inNodesFile + ", edgelist: " + inEdgesFile + ")\n"
        out += "alpha1,alpha2,correlation,p-value,vectorFile\n"
        f.write(out)

    #run the pipeline on all combos of alphas
    for a1 in alphaListFlt:
        for a2 in alphaListFlt:
            dstVectorFile = outVectorDir+"/vectors"+experimentName+"_"+str(a1)+"_"+str(a2)+".txt"
            subprocess.Popen(["./C++ code/main", inEdgesFile, dstVectorFile, str(a1), str(a2), repititions, simAllSeeds, inNodesFile]).wait() #run C++ code
            #vector_analysis.pearson_analysis(inNodesFile, dstVectorFile, outAnalysisFile, a1, a2)
            vector_analysis.knn(inNodesFile, dstVectorFile, outAnalysisFile, a1, a2, 3, 25)

    data_rep.heatmap(outAnalysisFile, outHeatMapFile)

if __name__=="__main__":
    main()
