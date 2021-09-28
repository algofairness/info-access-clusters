import argparse
import configparser
import seaborn as sns
import csv
import matplotlib.pyplot as plt
import statistics as stat
import networkx as nx
import csv 
from Fneighbors import *

#ConfigParser support https://docs.python.org/3/library/configparser.html#supported-ini-file-structure
parser = argparse.ArgumentParser(description='get path to the config file.')
parser.add_argument('filepath', help='path to the config file, including filename')
args = parser.parse_args()
configFile=args.filepath #get configFile
#configFile="PATH TO CONFIGFILE" #uncomment this line to overwrite configFile
config = configparser.ConfigParser() #make config object
print(config.sections())
config.read(configFile) #read configFile

#[OG]
pdf = config['OG']['pdf']
mean = config['OG']['mean']
meanNonSeeds = config['OG']['meanNonSeeds']
pNeighbors = config['OG']['pNeighbors']

#[FUNCTION]
funcNeighbors = config['FUNCTION']['funcNeighbors']
errorMargin = config['FUNCTION']['errorMargin']

#[FILES]
inSeedEdgesFile = config['FILES']['inSeedEdgesFile']
vectorsFile = config['FILES']['vectorsFile']

#[ORIGSETTINGS]
basic = config['ORIGSETTINGS']['basic']
SBM = config['ORIGSETTINGS']['SBM']
LFR = config['ORIGSETTINGS']['LFR']
removeNode = config['ORIGSETTINGS']['removeNode']
nodeList = config['ORIGSETTINGS']['nodeList']
edgeList = config['ORIGSETTINGS']['edgeList']
seedEdgelist = config['ORIGSETTINGS']['seedEdgelist']
size = int(config['ORIGSETTINGS']['size'])
intraNodes = float(config['ORIGSETTINGS']['intraNodes'])
interNodes = float(config['ORIGSETTINGS']['interNodes'])
aveDegree = int(config['ORIGSETTINGS']['aveDegree'])
maxDegree = int(config['ORIGSETTINGS']['maxDegree'])
mixingParameter = float(config['ORIGSETTINGS']['mixingParameter'])
probRemoveA = config['ORIGSETTINGS']['probRemoveA']
probRemoveB = config['ORIGSETTINGS']['probRemoveB']
numRemoveA = int(config['ORIGSETTINGS']['numRemoveA'])
numRemoveB = int(config['ORIGSETTINGS']['numRemoveB'])
probSeedA = config['ORIGSETTINGS']['probSeedA']
probSeedB = config['ORIGSETTINGS']['probSeedB']
numSeedA = config['ORIGSETTINGS']['numSeedA']
numSeedB = config['ORIGSETTINGS']['numSeedB']

def main():
    if pdf:
        pdf()
    if pNeighbors == 'yes':
        Pfunc_neighbors(funcNeighbors,vectorsFile,"twocommunities_neighborlist.txt",errorMargin,size*2)

def pdf():
    if meanNonSeeds == 'yes':
        Aseeds = []
        Bseeds = []
        #do this using list comprehension
        with open("../"+"twocommunities_seed_edgelist.txt") as file:
            edgelist = csv.reader(file,delimiter='\t')
            for line in edgelist:
                if line[0] == 's':
                    for node in line:
                        if node.isnumeric() and int(node) <= 1000:
                            Aseeds.append(node)
                        if node.isnumeric() and int(node)>1000:
                            Bseeds.append(node)
    with open("../"+"twocommunities_vectors.txt") as file:
        data = csv.reader(file, delimiter='\t')
        Aprob=[]
        Bprob=[]
        AnsProb = []
        BnsProb = []
        #print(Aseeds)
        for line in data:
            if int(line[0]) <= (1000): #add option when have removed from A
                Aprob.append(float(line[1]))
                if meanNonSeeds == 'yes':
                    if int(line[0]) not in Aseeds:
                        AnsProb.append(float(line[1]))
            else:
                Bprob.append(float(line[1]))
                if meanNonSeeds == 'yes':
                    if int(line[0]) not in Bseeds:
                        BnsProb.append(float(line[1]))
    if mean == 'yes':
        print("Mean prob of receiving info in community A:", stat.mean(Aprob))
        print("Mean prob of receiving info in community B:", stat.mean(Bprob))
    if meanNonSeeds == 'yes':
        print("Mean prob for non seed nodes of receiving info in community A:", stat.mean(AnsProb))
        print("Mean prob for non seed nodes of receiving info in community B:", stat.mean(BnsProb))        
    if removeNode == 'yes' and probRemoveA != None:
        if probSeedA:
            sns.distplot(Aprob,hist=True,kde=True,kde_kws = {'linewidth': 3},label='A community (PrA={} / PsA={})'.format(float(probRemoveA),float(probSeedA)),norm_hist=True)
            sns.distplot(Bprob,hist=True,kde=True,kde_kws = {'linewidth': 3},label='B community (PrB={} / PsB={})'.format(float(probRemoveB), float(probSeedB)),norm_hist=True)
            if basic == 'yes':
                plt.title("Two communities - prob of nodes removed and prob of node being a seed \n - prob of transmission: Intra={} / Inter={}".format(intraNodes,interNodes))
            if SBM == 'yes':
                plt.title("Two communities SBM - prob of nodes removed and prob of node being a seed \n - prob of transmission: Intra={} / Inter={}".format(intraNodes,interNodes))
            if LFR == 'yes':
                plt.title("Two communities LFR - prob of nodes removed and prob of node being a seed \n - k:{} / maxk: {} / mu:{}".format(aveDegree,maxDegree,mixingParameter))        
        if numSeedA:
            sns.distplot(Aprob,hist=True,kde=True,kde_kws = {'linewidth': 3},label='A community (PrA={} / NsA={})'.format(float(probRemoveA),numSeedA),norm_hist=True)
            sns.distplot(Bprob,hist=True,kde=True,kde_kws = {'linewidth': 3},label='B community (PrB={} / NsB={})'.format(float(probRemoveB),numSeedB),norm_hist=True)
            if basic == 'yes':
                plt.title("Two communities - prob of nodes removed and set number of seeds \n - prob of transmission: Intra={} / Inter={}".format(intraNodes,interNodes))
            if SBM == 'yes':
                plt.title("Two communities SBM - prob of nodes removed and set number of seeds \n - prob of transmission: Intra={} / Inter={}".format(intraNodes,interNodes))
            if LFR == 'yes':
                plt.title("Two communities LFR - prob of nodes removed and set number of seeds \n - k:{} / maxk: {} / mu:{}".format(aveDegree,maxDegree,mixingParameter)) 
    
    if removeNode == 'yes'and numRemoveA != None:
        if probSeedA:
            sns.distplot(Aprob,hist=True,kde=True,kde_kws = {'linewidth': 3},label='A community (NrA={} / PsA={})'.format(numRemoveA,float(probSeedA)),norm_hist=True)
            sns.distplot(Bprob,hist=True,kde=True,kde_kws = {'linewidth': 3},label='B community (NrB={} / PsB={})'.format(numRemoveB,float(probSeedB)),norm_hist=True)
            if basic == 'yes':
                plt.title("Two communities - set number of nodes removed and prob of node being a seed \n - prob of transmission: Intra={} / Inter={}".format(intraNodes,interNodes))
            if SBM == 'yes':
                plt.title("Two communities SBM - set number of nodes removed and prob of node being a seed \n - prob of transmission: Intra={} / Inter={}".format(intraNodes,interNodes))
            if LFR == 'yes':
                plt.title("Two communities LFR - set number of nodes removed and prob of node being a seed \n - k:{} / maxk: {} / mu:{}".format(aveDegree,maxDegree,mixingParameter)) 
        if numSeedA:
            sns.distplot(Aprob,hist=True,kde=True,kde_kws = {'linewidth': 3},label='A community (NrA=%d / NsA=%d)'%numRemoveA %numSeedA,norm_hist=True)
            sns.distplot(Bprob,hist=True,kde=True,kde_kws = {'linewidth': 3},label='B community (NrB=%d / NsB=%d)'%numRemoveB %numSeedB,norm_hist=True)
            if basic == 'yes':
                plt.title("Two communities - set number of nodes removed and set number of seeds \n - prob of transmission: Intra={} / Inter={}".format(intraNodes,interNodes))
            if SBM == 'yes':
                plt.title("Two communities SBM - set number of nodes removed and set number of seeds \n - prob of transmission: Intra={} / Inter={}".format(intraNodes,interNodes))
            if LFR == 'yes':
                plt.title("Two communities LFR - set number of nodes removed and set number of seeds \n - k:{} / maxk: {} / mu:{}".format(aveDegree,maxDegree,mixingParameter)) 
    else:
        if probSeedA:
            sns.distplot(Aprob,hist=True,kde=True,kde_kws = {'linewidth': 3},label='A community (PsA={}'.format(float(probSeedA)),norm_hist=True)
            sns.distplot(Bprob,hist=True,kde=True,kde_kws = {'linewidth': 3},label='B community (PsB={}'.format(float(probSeedB)),norm_hist=True) 
            if basic == 'yes':
                plt.title("Two communities - prob of node being a seed \n - prob of transmission: Intra={} / Inter={}".format(intraNodes,interNodes))
            if SBM == 'yes':
                plt.title("Two communities SBM - prob of node being a seed \n - prob of transmission: Intra={} / Inter={}".format(intraNodes,interNodes))            
            if LFR == 'yes':
                plt.title("Two communities LFR - prob of node being a seed \n - k:{} / maxk: {} / mu:{}".format(aveDegree,maxDegree,mixingParameter))                     
        if numSeedA:
            sns.distplot(Aprob,hist=True,kde=True,kde_kws = {'linewidth': 3},label='A community (NsA={}'.format(numSeedA),norm_hist=True)
            sns.distplot(Bprob,hist=True,kde=True,kde_kws = {'linewidth': 3},label='B community (NsB={}'.format(numSeedB),norm_hist=True)   
            if basic == 'yes':
                plt.title("Two communities - set number of seeds \n - prob of transmission: Intra={} /Inter={}".format(intraNodes,interNodes))
            if SBM == 'yes':
                plt.title("Two communities SBM - set number of seeds \n - prob of transmission: Intra={} /Inter={}".format(intraNodes,interNodes))       
            if LFR == 'yes':
                plt.title("Two communities LFR - set number of seeds \n - k:{} / maxk: {} / mu:{}".format(aveDegree,maxDegree,mixingParameter)) 
    plt.xlabel("Probability of node receiving information")
    plt.legend() 
    plt.show()

    

if __name__=="__main__":
    main()