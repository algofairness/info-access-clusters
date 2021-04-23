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
runAnalysis = config['GENERAL']['runAnalysis']
simAllSeeds = config['GENERAL']['simAllSeeds']
repititions = config['GENERAL']['repititions']
alphaListStr = config['GENERAL']['alphaList'] #must be changed to string of floats
alphaListFlt = [float(item) for item in alphaListStr.split(',')] #usable string of floats

#[FILES]
inEdgesFile = config['FILES']['inEdgesFile']
inNodesFile = config['FILES']['inNodesFile']
outputDir = config['FILES']['outputDir']
outVectorDir = config['FILES']['outVectorDir']
inVectorDir = config['FILES']['inVectorDir']
outAnalysisDir = config['FILES']['outAnalysisDir']
inAnalysisDir = config['FILES']['inAnalysisDir']

#[ANALYSIS]
knnNeighbors = config['ANALYSIS']['knnNeighbors']
knnRepititions = config['ANALYSIS']['knnRepititions']

def main():
    directories = make_directory(0) #directory for this specific experiment, always input 0
    expDir = directories[0]
    expAnalysisDir = directories[1]
    expVectorDir = directories[2] #this will change if generateVectors=='no'
    print("Created experiment directory... Paths are:")
    print("   Experiment directory:", expDir, "\n   Vector directory:", expVectorDir, "\n   Analysis directory:", expAnalysisDir)

    if generateVectors=="yes":
        print("starting simulation... vector files going to", expVectorDir)
        run_simulation(expVectorDir)
        print("simulation was run - vector files are in", expVectorDir)

    if runAnalysis=='yes':
        print("running analysis... files going to", expAnalysisDir)
        run_analysis(expVectorDir, expAnalysisDir)

        print("representing data from", expAnalysisDir, "... files going to", expAnalysisDir)
        run_datarep(expAnalysisDir, expAnalysisDir)


    if runAnalysis=='no':
        print("representing data from", inAnalysisDir, "... files going to", expAnalysisDir)
        run_datarep(inAnalysisDir, expAnalysisDir)

    return

#makes the directory structure
#makes/returns 3 directories/paths. the vector path changes based on generateVectors
def make_directory(versionNum):
    dirname = experimentName + "_v" + str(versionNum)
    dirPath = outputDir + dirname + "/"
    if os.path.isdir(dirPath): #if directory exsists...
        return make_directory(versionNum+1) #...recursively check for the next version
    else:
        os.mkdir(dirPath)
        os.mkdir(dirPath+outAnalysisDir)
        analysisPath = dirPath+outAnalysisDir
        if generateVectors=='yes':
            os.mkdir(dirPath+outVectorDir)
            vectorPath = dirPath+outVectorDir
        if generateVectors=='no':
            vectorPath = inVectorDir

        return dirPath, analysisPath, vectorPath

def run_simulation(vectorDir):
    for a1 in alphaListFlt:
        for a2 in alphaListFlt:
            print(experimentName, vectorDir, str(a1))
            outVectorFile = vectorDir+"vectors"+experimentName+"_"+str(a1)+"_"+str(a2)+"_.txt"
            subprocess.Popen(["./C++ code/main", inEdgesFile, outVectorFile, str(a1), str(a2), repititions, simAllSeeds, inNodesFile]).wait() #run C++ code
    return 1

def run_analysis(vectorDir, analysisDir):
    analysisFile = analysisDir+"analysis.txt"
    with open(analysisFile, 'a') as f: #make analysis file header
        out = "EXPERIMENT: " + experimentName + "(nodeList: " + inNodesFile + ", edgelist: " + inEdgesFile + ")\n"
        out += "alpha1,alpha2,correlation,p-value,vectorFile\n"
        f.write(out)
    #go through vectorDir, run analysis on each vector file
    for file in os.scandir(vectorDir):
        if file.is_file() and file.name.endswith('.txt'):
            alphas=get_alphas_from_filepath(file.path)
            vector_analysis.knn(inNodesFile, file.path, analysisFile, alphas[0], alphas[1], int(knnNeighbors), int(knnRepititions))
            #vector_analysis.pearson_analysis(inNodesFile, file.path, analysisFile, alphas[0], alphas[1])
    return 1

def run_datarep(inAnalysisDir, outAnalysisDir):
    inAnalysisFile= inAnalysisDir+"analysis.txt"
    outHeatmapFile= outAnalysisDir+"heatmap.png"
    data_rep.heatmap(inAnalysisFile, outHeatmapFile)
    return

def get_alphas_from_filepath(filepath):
    pathlist=filepath.split('_')
    alpha1 = pathlist[-3]
    alpha2 = pathlist[-2]
    return alpha1, alpha2

if __name__=="__main__":
    main()
