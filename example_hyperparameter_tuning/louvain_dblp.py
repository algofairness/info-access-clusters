"""DBLP-specific wrappers for mp.louvain_preprocess() pipeline."""
import main_pipelines as mp


# IDENTIFIER_STRING
# INPUT_PICKLED_GRAPH
# run_elbow
# K

def main():
    mp.IDENTIFIER_STRING = "dblp"
    mp.INPUT_PICKLED_GRAPH = "output_files/main_files/{}_pickle".format(mp.IDENTIFIER_STRING)
    mp.K = 2
    mp.LABELING_FILE = "output_files/main_files/{}_K{}_labeling_file_louvain.csv".format(mp.IDENTIFIER_STRING, mp.K)
    mp.EXPERIMENT = "louvain"
    mp.louvain_preprocess()
    # mp.statistical_analyses()
    return


if __name__ == "__main__":
    main()
