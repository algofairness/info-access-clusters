"""DBLP-specific wrappers for mp.count_cc() and experimentation pipeline."""
import main_pipelines as mp

# Main variable input:
METHOD = "spectral"

def main():
    mp.IDENTIFIER_STRING = "dblp"
    mp.INPUT_PICKLED_GRAPH = "output_files/main_files/{}_pickle".format(mp.IDENTIFIER_STRING)
    mp.K = 2
    mp.LABELING_FILE = "output_files/main_files/{}_K{}_labeling_file_{}.csv".format(mp.IDENTIFIER_STRING, mp.K, METHOD)
    mp.ALPHA_VALUES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    mp.count_cc_wrapper()
    return


if __name__ == '__main__':
    main()

    #####
    # cc cluster_sizes match new graphs sizes
    # cc cluster_sizes match democrat and le_score cluster_sizes
    # cc_cs = mp.cc_cluster_sizes("output_files/strong-house_K2_cc_iac.csv")
    # new_cs = mp.cs_by_search_unnamed("output_files/name_edited_strong-house_K2_output_strings_le_score.txt", ALPHA_VALUES)
    #
    #####
