"""DBLP-specific wrappers for mp.core_periphery() and experimentation pipeline."""
import main_pipelines as mp


def main():
    mp.IDENTIFIER_STRING = "dblp"
    mp.INPUT_PICKLED_GRAPH = "output_files/main_files/{}_pickle".format(mp.IDENTIFIER_STRING)
    mp.K = 2
    # mp.CP_THRESHOLD = -1
    mp.LABELING_FILE = "output_files/main_files/{}_K2_labeling_file_cp.csv".format(mp.IDENTIFIER_STRING)
    mp.EXPERIMENT = "cp"
    # mp.core_periphery()
    mp.statistical_analyses()
    return


if __name__ == '__main__':
    main()
