"""DBLP-specific wrappers for mp.iac_vs_x_ari() for fluidcr and experimentation pipeline."""
import main_pipelines as mp


def main():
    mp.IDENTIFIER_STRING = "dblp"
    mp.INPUT_PICKLED_GRAPH = "output_files/main_files/{}_pickle".format(mp.IDENTIFIER_STRING)
    mp.K = 2
    mp.IAC_LABELING_FILE = "output_files/main_files/{}_K{}_labeling_file_iac.csv".format(mp.IDENTIFIER_STRING, mp.K)

    main_labeling_file = "output_files/fluidcr/dblp_K2_labeling_file_fluidcr.csv"
    mp.preprocess_fluidcr(main_labeling_file)

    for seed in mp.SEEDS:
        mp.LABELING_FILE = "output_files/fluidcr/{}_labeling_files_fluidcr/{}_K{}_labeling_file_fluidcrs{}.csv".format(
            mp.IDENTIFIER_STRING, mp.IDENTIFIER_STRING, mp.K, seed)
        mp.EXPERIMENT = "fluidcrs{}".format(seed)
        mp.ALPHA_VALUES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        mp.iac_vs_x_ari()
        mp.statistical_analyses()
    return


if __name__ == '__main__':
    main()
