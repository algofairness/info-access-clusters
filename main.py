import main_pipelines as mp
import ctypes
from pathlib import Path
import subprocess
import sys
import os
import argparse
import configparser
import json
import shutil
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
runDataRep = config['GENERAL']['runDataRep']
runHoldout = config['GENERAL']['runHoldout']
genHoldVectors = config['GENERAL']['genHoldVectors']
simAllSeeds = config['GENERAL']['simAllSeeds']
repititions = config['GENERAL']['repititions']
alphaListStr = config['GENERAL']['alphaList'] #must be changed to string of floats
alphaListFlt = [float(item) for item in alphaListStr.split(',')] #usable string of floats
alpha1list = config['GENERAL']['alpha1list']
alpha1listFlt = [float(item) for item in alpha1list.split(',')] #usable string of floats
alpha2list = config['GENERAL']['alpha2list']
alpha2listFlt = [float(item) for item in alpha2list.split(',')] #usable string of floats

#[FILES]
inEdgesFile = config['FILES']['inEdgesFile']
inNodesFile = config['FILES']['inNodesFile']
inHoldEdgesFile = config['FILES']['inHoldEdgesFile']
inHoldNodesFile = config['FILES']['inHoldNodesFile']
outputDir = config['FILES']['outputDir']
outVectorDir = config['FILES']['outVectorDir']
outHoldVecDir = config['FILES']['outHoldVecDir']
inVectorDir = config['FILES']['inVectorDir']
inHoldVecDir = config['FILES']['inHoldVecDir']
outAnalysisDir = config['FILES']['outAnalysisDir']
outHoldAnalysisDir = config['FILES']['outHoldAnalysisDir']
inAnalysisDir = config['FILES']['inAnalysisDir']
inHoldAnalysisDir = config['FILES']['inHoldAnalysisDir']

#[ANALYSIS]
vsDummy = config['ANALYSIS']['vsDummy']
usePCA = config['ANALYSIS']['usePCA']
useZachKNN = config['ANALYSIS']['useZachKNN']
useKNN = config['ANALYSIS']['useKNN']
useSVR = config['ANALYSIS']['useSVR']
useRandomForest = config['ANALYSIS']['useRandomForest']
knnNeighbors = config['ANALYSIS']['knnNeighbors']
knnRepititions = config['ANALYSIS']['knnRepititions']
pcaComponents = config['ANALYSIS']['pcaComponents']

def main():
    directories = make_directory(0) #directory for this specific experiment, always input 0
    expDir = directories[0]
    expAnalysisDir = directories[1]
    expVectorDir = directories[2] #this will change if generateVectors=='no'
    if runHoldout=='yes':
        expHoldAnalysisDir = directories[3]
        expHoldVectorDir = directories[4] #this will change if generateVectors=='no'

    print("Created experiment directory... Paths are:")
    print("   Experiment directory:", expDir, "\n   Vector directory:", expVectorDir, "\n   Analysis directory:", expAnalysisDir)

    if runHoldout=='yes':
        #run_holdout_analysis(expHoldVectorDir, expHoldAnalysisDir, inHoldNodesFile)
        run_holdout_pipeline(directories)
        return

    if generateVectors=="yes":
        print("starting simulation... vector files going to", expVectorDir)
        run_simulation(expVectorDir)
        print("simulation was run - vector files are in", expVectorDir)

    if genHoldVectors=='yes':
        print("starting holdout simulation... vector files going to", expHoldVectorDir)
        run_holdout_simulation(expHoldVectorDir)
        print("simulation was run - vector files are in", expHoldVectorDir)

    if runAnalysis=='yes':
        print("running analysis... files going to", expAnalysisDir)
        run_analysis(expVectorDir, expAnalysisDir, inNodesFile)

        if runDataRep=='yes':
            print("representing data from", expAnalysisDir, "... files going to", expAnalysisDir)
            run_datarep(expAnalysisDir, expAnalysisDir)

    if runAnalysis=='no':
        if runDataRep=='yes':
            print("representing data from", inAnalysisDir, "... files going to", expAnalysisDir)
            run_datarep(inAnalysisDir, expAnalysisDir)

    return

#makes the directory structure with a copy of the configFile in it
#makes/returns 3 directories/paths. the vector path changes based on generateVectors
def make_directory(versionNum):
    dirname = experimentName + "_v" + str(versionNum)
    dirPath = outputDir + dirname + "/"
    if os.path.isdir(dirPath): #if directory exsists...
        return make_directory(versionNum+1) #...recursively check for the next version
    else:
        os.mkdir(dirPath) #make the experiment directory

        configCopy = dirPath + experimentName + "ConfigRecord.ini"
        shutil.copyfile(configFile, configCopy) #copy config file to experiment folder

        #make analysis directories
        if runHoldout == 'yes':
            holdAnalysisPath = dirPath+outHoldAnalysisDir
            os.mkdir(holdAnalysisPath)
        analysisPath = dirPath+outAnalysisDir
        os.mkdir(analysisPath)

        #make vector directories
        if generateVectors=='yes':
            if genHoldVectors=='yes':
                holdVectorPath = dirPath+outHoldVecDir
                os.mkdir(holdVectorPath)
            if genHoldVectors=='no':
                holdVectorPath = dirPath+inHoldVecDir
                os.mkdir(holdVectorPath)
            vectorPath = dirPath+outVectorDir
            os.mkdir(vectorPath)
        if generateVectors=='no':
            if genHoldVectors=='yes':
                holdVectorPath = dirPath+outHoldVecDir
                os.mkdir(holdVectorPath)
            if genHoldVectors=='no':
                holdVectorPath = dirPath+inHoldVecDir
            vectorPath = inVectorDir

        #return paths
        if runHoldout == 'yes':
            return dirPath, analysisPath, vectorPath, holdAnalysisPath, holdVectorPath
        if runHoldout == 'no':
            return dirPath, analysisPath, vectorPath

def run_simulation(vectorDir):
    for a1 in alphaListFlt:
        for a2 in alphaListFlt:
            #run normal simulation (always happens)
            outVectorFile = vectorDir+"vectors"+experimentName+"_"+str(a1)+"_"+str(a2)+"_.txt"
            subprocess.Popen(["./C++ code/main", inEdgesFile, outVectorFile,
                            str(a1), str(a2), repititions, simAllSeeds, inNodesFile]).wait() #run C++ code
    return 1


def run_holdout_simulation(vectorDir):
    for a1 in alphaListFlt:
        for a2 in alphaListFlt:
            outVectorFile = vectorDir+"holdoutVectors"+experimentName+"_"+str(a1)+"_"+str(a2)+"_.txt"
            subprocess.Popen(["./C++ code/main", inHoldEdgesFile, outVectorFile,
                            str(a1), str(a2), repititions, simAllSeeds, inHoldNodesFile]).wait() #run C++ code
    return 1


def run_holdout_pipeline(directories):
    expDir = directories[0]
    expAnalysisDir = directories[1]
    expVectorDir = directories[2] #this will change if generateVectors=='no'
    expHoldAnalysisDir = directories[3]
    expHoldVectorDir = directories[4]
    completeAnalysisFile = expDir+"completeAnalysis"+experimentName+".txt"
    components = int(pcaComponents)

    with open(completeAnalysisFile, 'a') as f:
        header = "a1,a2,mseRegDummy,stdRegDummy,mseHoldDummy,"
        header += "mseRegKNN,stdRegKNN,mseHoldKNN,"
        header += "mseRegSVR,stdRegSVR,mseHoldSVR,"
        header += "mseRegRF,stdRegRF,mseHoldRF\n"
        f.write(header)

    for a1 in alpha1listFlt:
        for a2 in alpha2listFlt:
            #run normal simulation
            outVectorFile = expVectorDir+"vectors"+experimentName+"_"+str(a1)+"_"+str(a2)+"_.txt"
            subprocess.Popen(["./C++ code/main", inEdgesFile, outVectorFile,
                            str(a1), str(a2), repititions, simAllSeeds, inNodesFile]).wait() #run C++ code
            #run holdout simulation
            outHoldVectorFile = expHoldVectorDir+"holdoutVectors"+experimentName+"_"+str(a1)+"_"+str(a2)+"_.txt"
            subprocess.Popen(["./C++ code/main", inHoldEdgesFile, outHoldVectorFile,
                            str(a1), str(a2), repititions, simAllSeeds, inHoldNodesFile]).wait() #run C++ code

            #run analysis
            #REG DUMMY
            regAnalysisDummy = expAnalysisDir+"regAnalysisDummy.txt"
            header="alpha1,alpha2,mean,vectorFile\n"
            make_analysis_file(regAnalysisDummy, header)
            regDummy=vector_analysis.runDummy(inNodesFile, outVectorFile, regAnalysisDummy, a1, a2)

            #HOLD DUMMY
            holdAnalysisDummy = expAnalysisDir+"holdAnalysisDummy.txt"
            header="alpha1,alpha2,mean,vectorFile\n"
            make_analysis_file(holdAnalysisDummy, header)
            holdDummy=vector_analysis.holdoutDummy(inNodesFile, outVectorFile, inHoldNodesFile,
                                                   outHoldVectorFile, holdAnalysisDummy, a1, a2, components)

        #if useKNN == 'yes':
            #regular KNN
            analysisFile = expAnalysisDir+"analysisKNN.txt"
            header="alpha1,alpha2,mse,vectorFile\n"
            make_analysis_file(analysisFile, header)
            regKNN=vector_analysis.KNN(inNodesFile, outVectorFile, analysisFile, a1,
                                a2, int(knnNeighbors), int(knnRepititions))

            #holdout KNN
            holdAnalysisFile = expHoldAnalysisDir+"analysisHoldoutKNN.txt"
            header="alpha1,alpha2,accuracy,vectorFile\n"
            make_analysis_file(holdAnalysisFile, header)
            holdKNN=vector_analysis.holdoutKNN(inNodesFile, outVectorFile, inHoldNodesFile,
                                               outHoldVectorFile, holdAnalysisFile, a1, a2, int(knnNeighbors), components)

        #if useSVR == 'yes':
            #normal SVR
            analysisFile = expAnalysisDir+"analysisSVR.txt"
            header="alpha1,alpha2,mean,std,vectorFile\n" #come back
            make_analysis_file(analysisFile, header)
            regSVR=vector_analysis.runSVR(inNodesFile, outVectorFile, analysisFile, a1, a2)

            #holdout SVR
            holdAnalysisFile = expHoldAnalysisDir+"analysisHoldoutSVR.txt"
            header="alpha1,alpha2,accuracy,vectorFile\n"
            make_analysis_file(holdAnalysisFile, header)
            holdSVR=vector_analysis.holdoutSVR(inNodesFile, outVectorFile, inHoldNodesFile,
                                       outHoldVectorFile, holdAnalysisFile, a1, a2, components)

        #if useRandomForest == 'yes':
            #normal RF
            analysisFile = expAnalysisDir+"analysisRandomForest.txt"
            header="alpha1,alpha2,mean,std,vectorFile\n" #come back
            make_analysis_file(analysisFile, header)
            regRF=vector_analysis.randomForest(inNodesFile, outVectorFile, analysisFile, a1, a2)

            #holdout RF
            holdAnalysisFile = expHoldAnalysisDir+"analysisHoldoutRandomForest.txt"
            header="alpha1,alpha2,accuracy,vectorFile\n"
            make_analysis_file(holdAnalysisFile, header)
            holdRF=vector_analysis.holdoutRandomForest(inNodesFile, outVectorFile, inHoldNodesFile,
                                       outHoldVectorFile, holdAnalysisFile, a1, a2, components)

            with open(completeAnalysisFile, 'a') as f:
                out = str(a1) + ',' + str(a2) + ','
                out += str(round(regDummy[0],3))+','+str(round(regDummy[1],3))+','+ str(round(holdDummy,3))+','
                out += str(round(regKNN[0],3))+','+str(round(regKNN[1],3))+','+ str(round(holdKNN,3))+ ','
                out += str(round(regSVR[0],3))+','+str(round(regSVR[1],3))+','+ str(round(holdSVR,3))+ ','
                out += str(round(regRF[0],3))+','+str(round(regRF[1],3))+','+ str(round(holdRF,3))+'\n'
                f.write(out)
            #write to file

    return 1



def run_analysis(vectorDir, analysisDir, nodefile):

    if vsDummy == 'yes':
        print("Running Dummy analysis...")
        analysisFile = analysisDir+"analysisDummy.txt"
        header="alpha1,alpha2,mean,std,vectorFile\n" #come back
        make_analysis_file(analysisFile, header)
        for file in os.scandir(vectorDir):
            if file.is_file() and file.name.endswith('.txt'):
                alphas=get_alphas_from_filepath(file.path)
                vector_analysis.runDummy(nodefile, file.path, analysisFile, alphas[0], alphas[1])

    #copy and change below three lines when adding new analysis
    if usePCA == 'yes':
        print("Running PCA analysis...")
        analysisFile = analysisDir+"analysisPCA.txt"
        header="alpha1,alpha2,correlation,p-value,vectorFile\n"
        make_analysis_file(analysisFile, header)
        #go through vectorDir, run analysis on each vector file
        for file in os.scandir(vectorDir):
            if file.is_file() and file.name.endswith('.txt'):
                alphas=get_alphas_from_filepath(file.path)
                #change below line when adding new analysis
                vector_analysis.pearson_analysis(nodefile, file.path,
                 analysisFile, alphas[0], alphas[1])

    if useZachKNN == 'yes':
        print("Running zachKNN analysis...")
        analysisFile = analysisDir+"analysiszachKNN.txt"
        header="alpha1,alpha2,accuracy,vectorFile\n"
        make_analysis_file(analysisFile, header)
        for file in os.scandir(vectorDir):
            if file.is_file() and file.name.endswith('.txt'):
                alphas=get_alphas_from_filepath(file.path)
                vector_analysis.zachKNN(nodefile, file.path, analysisFile, alphas[0],
                 alphas[1], int(knnNeighbors), int(knnRepititions))

    if useKNN == 'yes':
        analysisFile = analysisDir+"analysisKNN.txt"
        header="alpha1,alpha2,accuracy,std,vectorFile\n"
        make_analysis_file(analysisFile, header)
        for file in os.scandir(vectorDir):
            if file.is_file() and file.name.endswith('.txt'):
                alphas=get_alphas_from_filepath(file.path)
                vector_analysis.KNN(nodefile, file.path, analysisFile, alphas[0],
                 alphas[1], int(knnNeighbors), int(knnRepititions))

    if useRandomForest == 'yes':
        print("Running Random Forest analysis...")
        analysisFile = analysisDir+"analysisRandomForest.txt"
        header="alpha1,alpha2,mean,std,vectorFile\n" #come back
        make_analysis_file(analysisFile, header)
        for file in os.scandir(vectorDir):
            if file.is_file() and file.name.endswith('.txt'):
                alphas=get_alphas_from_filepath(file.path)
                vector_analysis.randomForest(nodefile, file.path, analysisFile, alphas[0], alphas[1])

    if useSVR == 'yes':
        print("Running SVR analysis...")
        analysisFile = analysisDir+"analysisSVR.txt"
        header="alpha1,alpha2,mean,std,vectorFile\n" #come back
        make_analysis_file(analysisFile, header)
        for file in os.scandir(vectorDir):
            if file.is_file() and file.name.endswith('.txt'):
                alphas=get_alphas_from_filepath(file.path)
                vector_analysis.runSVR(nodefile, file.path, analysisFile, alphas[0], alphas[1])
    return 1


def run_holdout_analysis(vectorDir, analysisDir, nodefile):

    if useKNN == 'yes':
        analysisFile = analysisDir+"analysisHoldoutKNN.txt"
        header="alpha1,alpha2,accuracy,vectorFile\n"
        make_analysis_file(analysisFile, header)
        for file in os.scandir(vectorDir):
            if file.is_file() and file.name.endswith('.txt'):
                alphas=get_alphas_from_filepath(file.path)
                vector_analysis.holdoutKNN(nodefile, file.path, analysisFile, alphas[0],
                 alphas[1], int(knnNeighbors), int(knnRepititions))

    if useRandomForest == 'yes':
        print("Running Holdout Random Forest analysis...")
        analysisFile = analysisDir+"analysisHoldoutRandomForest.txt"
        header="alpha1,alpha2,mean,std,vectorFile\n" #come back
        make_analysis_file(analysisFile, header)
        for file in os.scandir(vectorDir):
            if file.is_file() and file.name.endswith('.txt'):
                alphas=get_alphas_from_filepath(file.path)
                vector_analysis.holdoutRandomForest(nodefile, file.path, analysisFile, alphas[0], alphas[1])

    if useSVR == 'yes':
        print("Running Holdout SVR analysis...")
        analysisFile = analysisDir+"analysisHoldoutSVR.txt"
        header="alpha1,alpha2,mean,std,vectorFile\n" #come back
        make_analysis_file(analysisFile, header)
        for file in os.scandir(vectorDir):
            if file.is_file() and file.name.endswith('.txt'):
                alphas=get_alphas_from_filepath(file.path)
                vector_analysis.holdoutSVR(nodefile, file.path, analysisFile, alphas[0], alphas[1])
    return 1


def make_analysis_file(analysisFile, header):
    with open(analysisFile, 'a') as f: #make analysis file header
        out = "EXPERIMENT: " + experimentName + "\n"
        out += header
        f.write(out)
    return 1

def run_datarep(inAnalysisDir, outAnalysisDir):
    if usePCA == 'yes':
        inAnalysisFile= inAnalysisDir+"analysisPCA.txt"
        outHeatmapFile= outAnalysisDir+"heatmapPCA.png"
        data_rep.pcaHeatmap(inAnalysisFile, outHeatmapFile)
        if vsDummy == 'yes':
            boop = 'bop' #do nothing
    if useZachKNN == 'yes':
        inAnalysisFile= inAnalysisDir+"analysiszachKNN.txt"
        outHeatmapFile= outAnalysisDir+"heatmapzachKNN.png"
        data_rep.zachKNNHeatmap(inAnalysisFile, outHeatmapFile)
        if vsDummy == 'yes':
            analysisName = 'ZachKNN'
            inDummyFile= inAnalysisDir+"analysisDummy.txt"
            outVsDummyFile= outAnalysisDir+"heatmapZachKNNvsDummy.png"
            data_rep.vsDummyHeatmap(analysisName, inAnalysisFile, inDummyFile, outVsDummyFile)
    if useKNN == 'yes':
        inAnalysisFile= inAnalysisDir+"analysisKNN.txt"
        outHeatmapFile= outAnalysisDir+"heatmapKNN.png"
        data_rep.KNNHeatmap(inAnalysisFile, outHeatmapFile)
        if vsDummy == 'yes':
            analysisName = 'KNN'
            inDummyFile= inAnalysisDir+"analysisDummy.txt"
            outVsDummyFile= outAnalysisDir+"heatmapKNNvsDummy.png"
            data_rep.vsDummyHeatmap(analysisName, inAnalysisFile, inDummyFile, outVsDummyFile)

    if useRandomForest == 'yes':
        inAnalysisFile= inAnalysisDir+"analysisRandomForest.txt"
        outHeatmapFile= outAnalysisDir+"heatmapRandomForest.png"
        data_rep.randomForestHeatmap(inAnalysisFile, outHeatmapFile)
        if vsDummy == 'yes':
            analysisName = 'RandomForest'
            inDummyFile= inAnalysisDir+"analysisDummy.txt"
            outVsDummyFile= outAnalysisDir+"heatmapRandomForestvsDummy.png"
            data_rep.vsDummyHeatmap(analysisName, inAnalysisFile, inDummyFile, outVsDummyFile)

    if useSVR == 'yes':
        inAnalysisFile= inAnalysisDir+"analysisSVR.txt"
        outHeatmapFile= outAnalysisDir+"heatmapSVR.png"
        data_rep.SVRHeatmap(inAnalysisFile, outHeatmapFile)
        if vsDummy == 'yes':
            analysisName = 'SVR'
            inDummyFile= inAnalysisDir+"analysisDummy.txt"
            outVsDummyFile= outAnalysisDir+"heatmapSVRvsDummy.png"
            data_rep.vsDummyHeatmap(analysisName, inAnalysisFile, inDummyFile, outVsDummyFile)

    if vsDummy == 'yes':
        inAnalysisFile= inAnalysisDir+"analysisDummy.txt"
        outHeatmapFile= outAnalysisDir+"heatmapDummy.png"
        data_rep.dummyHeatmap(inAnalysisFile, outHeatmapFile)
    return

def get_alphas_from_filepath(filepath):
    pathlist=filepath.split('_')
    alpha1 = pathlist[-3]
    alpha2 = pathlist[-2]
    return alpha1, alpha2

if __name__=="__main__":
    main()
