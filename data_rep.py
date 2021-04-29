#implementation from https://www.python-graph-gallery.com/heatmap/ and
#https://www.kite.com/python/docs/seaborn.heatmap
#https://blog.quantinsti.com/creating-heatmap-using-python-seaborn/
#https://towardsdatascience.com/heatmap-basics-with-pythons-seaborn-fb92ea280a6c
import seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#datafile = "output_files/analysis/analysis1-5.txt"
#outfile = "output_files/analysis/heatmap1-5.png"

def main():
    #heatmap(datafile, outfile)
    return 1

#TO DO: make one heatmap file and just pass in analysis name

def pcaHeatmap(datafile, outfile):
    df = pd.read_csv(datafile, header=1, usecols=[0,1,2,3])
    shapedDF = df.pivot(index='alpha2', columns='alpha1', values='correlation')
    print(shapedDF)
    fig, ax = plt.subplots(figsize=(11, 9))
    # plot heatmap
    seaborn.heatmap(shapedDF, cmap="Blues", linewidth=0.3,
                    cbar_kws={"shrink": .8}).invert_yaxis()
    title = 'PCA Rank Correlations\n'.upper()
    plt.title(title, loc='left')
    plt.savefig(outfile)
    return

def zachKNNHeatmap(datafile, outfile):
    df = pd.read_csv(datafile, header=1, usecols=[0,1,2])
    shapedDF = df.pivot(index='alpha2', columns='alpha1', values='accuracy')
    print(shapedDF)
    fig, ax = plt.subplots(figsize=(11, 9))
    # plot heatmap
    seaborn.heatmap(shapedDF, cmap="Blues", linewidth=0.3,
                    cbar_kws={"shrink": .8}).invert_yaxis()
    title = 'Zach KNN Average Accuracy\n'.upper()
    plt.title(title, loc='left')
    plt.savefig(outfile)
    return

def KNNHeatmap(datafile, outfile):
    df = pd.read_csv(datafile, header=1, usecols=[0,1,2])
    shapedDF = df.pivot(index='alpha2', columns='alpha1', values='accuracy')
    print(shapedDF)
    fig, ax = plt.subplots(figsize=(11, 9))
    # plot heatmap
    seaborn.heatmap(shapedDF, cmap="Blues", linewidth=0.3,
                    cbar_kws={"shrink": .8}).invert_yaxis()
    title = 'KNN Average Mean Squared Error\nAcross K Folds\n'.upper()
    plt.title(title, loc='left')
    plt.savefig(outfile)
    return

def randomForestHeatmap(datafile, outfile):
    df = pd.read_csv(datafile, header=1, usecols=[0,1,2,3])
    shapedDF = df.pivot(index='alpha2', columns='alpha1', values='mean')
    print(shapedDF)
    fig, ax = plt.subplots(figsize=(11, 9))
    # plot heatmap
    seaborn.heatmap(shapedDF, cmap="Blues", linewidth=0.3,
                    cbar_kws={"shrink": .8}).invert_yaxis()
    title = 'Random Forests Average Mean Squared Error\nAcross K Folds\n'.upper()
    plt.title(title, loc='left')
    plt.savefig(outfile)
    return

def SVRHeatmap(datafile, outfile):
    df = pd.read_csv(datafile, header=1, usecols=[0,1,2,3])
    shapedDF = df.pivot(index='alpha2', columns='alpha1', values='mean')
    print(shapedDF)
    fig, ax = plt.subplots(figsize=(11, 9))
    # plot heatmap
    seaborn.heatmap(shapedDF, cmap="Blues", linewidth=0.3,
                    cbar_kws={"shrink": .8}).invert_yaxis()
    title = 'SVR Average Mean Squared Error Across K Folds\n'.upper()
    plt.title(title, loc='left')
    plt.savefig(outfile)
    return

def dummyHeatmap(datafile, outfile):
    df = pd.read_csv(datafile, header=1, usecols=[0,1,2,3])
    shapedDF = df.pivot(index='alpha2', columns='alpha1', values='mean')
    print(shapedDF)
    fig, ax = plt.subplots(figsize=(11, 9))
    # plot heatmap
    seaborn.heatmap(shapedDF, cmap="Blues", linewidth=0.3,
                    cbar_kws={"shrink": .8}).invert_yaxis()
    title = 'Dummy Average Mean Squared Error Across K Folds\n'.upper()
    plt.title(title, loc='left')
    plt.savefig(outfile)
    return

def vsDummyHeatmap(analysisName, realfile, dummyfile, outfile):
    real_df = pd.read_csv(realfile, header=1, usecols=[0,1,2])
    dummy_df = pd.read_csv(dummyfile, header=1, usecols=[0,1,2])
    if analysisName == 'PCA':
        return
    if (analysisName == 'ZachKNN') or (analysisName == 'KNN'):
        real_df['accuracy'] = dummy_df['mean']/real_df['accuracy']
        shaped_result = real_df.pivot(index='alpha2', columns='alpha1', values='accuracy')
        print(shaped_result)
        fig, ax = plt.subplots(figsize=(11, 9))
        seaborn.heatmap(shaped_result, cmap="Blues", linewidth=0.3,
                        cbar_kws={"shrink": .8}).invert_yaxis()
        title = analysisName+' Average Mean Squared Error \nDivided by Dummy Mean Squared Error\n'.upper()
        plt.title(title, loc='left')
        plt.savefig(outfile)
    if (analysisName == 'RandomForest') or (analysisName == 'SVR'):
        real_df['mean'] = dummy_df['mean']/real_df['mean']
        shaped_result = real_df.pivot(index='alpha2', columns='alpha1', values='mean')
        print(shaped_result)
        fig, ax = plt.subplots(figsize=(11, 9))
        seaborn.heatmap(shaped_result, cmap="Blues", linewidth=0.3,
                        cbar_kws={"shrink": .8}).invert_yaxis()
        title = analysisName+' Average Mean Squared Error \nDivided by Dummy Mean Squared Error\n'.upper()
        plt.title(title, loc='left')
        plt.savefig(outfile)
    else:
        print("unable to make vsDummy heatmap for", analysisName)
    return

main()
