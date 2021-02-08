"""DBLP-specific wrappers for mp.iac_vs_x_ari() and experimentation pipeline."""
import main_pipelines as mp

# Main variable input:
METHOD = "cp"


def main():
    mp.IDENTIFIER_STRING = "dblp"
    mp.INPUT_PICKLED_GRAPH = "output_files/main_files/{}_pickle".format(mp.IDENTIFIER_STRING)
    mp.K = 2
    mp.IAC_LABELING_FILE = "output_files/main_files/{}_K{}_labeling_file_iac.csv".format(mp.IDENTIFIER_STRING, mp.K)
    mp.LABELING_FILE = "output_files/main_files/{}_K{}_labeling_file_{}.csv".format(mp.IDENTIFIER_STRING, mp.K, METHOD)
    mp.ALPHA_VALUES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    mp.iac_vs_x_ari()
    return


if __name__ == '__main__':
    main()
