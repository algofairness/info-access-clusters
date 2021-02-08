"""DBLP-specific wrappers for mp.repeated_fluidc() pipeline."""
import main_pipelines as mp


def main():
    mp.IDENTIFIER_STRING = "dblp"
    mp.INPUT_PICKLED_GRAPH = "output_files/main_files/{}_pickle".format(mp.IDENTIFIER_STRING)
    mp.K = 2
    mp.LABELING_FILE = "output_files/main_files/{}_K{}_labeling_file_fluidc.csv".format(mp.IDENTIFIER_STRING, mp.K)
    mp.EXPERIMENT = "fluidc"
    mp.repeated_fluidc()
    return


if __name__ == "__main__":
    main()
