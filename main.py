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

def main():

    parser = argparse.ArgumentParser(description='get path to the config file.')
    parser.add_argument('filepath', help='path to the config file, including filename')
    args = parser.parse_args()
    configFile=args.filepath
    #configFile='config.ini'
    #ConfigParser support https://docs.python.org/3/library/configparser.html#supported-ini-file-structure
    config = configparser.ConfigParser()
    config.read(configFile)
    #global experiment variables:
    expName = config['GENERAL']['experimentName']
    srcEdges = config['GENERAL']['srcEdgeListFile']
    srcNodes = config['GENERAL']['srcNodeListFile']
    dstVectorDir = config['GENERAL']['dstVectorDir']
    dstAnalysisFile = config['GENERAL']['dstAnalysisFile']
    dstHeatMapFile = config['GENERAL']['dstHeatMapFile']

    repNumber = config['GENERAL']['repititions']
    simSeeds = config['GENERAL']['simAllSeeds']

    #alpha1 = config['GENERAL']['alpha1']
    #alpha2 = config['GENERAL']['alpha2']

    with open(dstAnalysisFile, 'a') as f:
        out = "EXPERIMENT: " + expName + "(nodeList: " + srcNodes + ", edgelist: " + srcEdges + ")\n"
        out += "alpha1,alpha2,correlation,p-value,vectorFile\n"
        f.write(out)


    #run the pipeline on all combos of alphas
    #alphalist = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]
    alphalistStr = config['GENERAL']['alphalist']
    alphalistFlt = [float(item) for item in alphalistStr.split(',')]

    for a1 in alphalistFlt:
        for a2 in alphalistFlt:
            dstVectorFile = dstVectorDir+"/vectors"+expName+"_"+str(a1)+"_"+str(a2)+".txt"
            subprocess.Popen(["./C++ code/main", srcEdges, dstVectorFile, str(a1), str(a2), repNumber, simSeeds, srcNodes]).wait() #run C++ code
            vector_analysis.pearson_analysis(srcNodes, dstVectorFile, dstAnalysisFile, a1, a2)

    data_rep.heatmap(dstAnalysisFile, dstHeatMapFile)

if __name__=="__main__":
    main()
