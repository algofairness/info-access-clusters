"""Pipeline to count the mean adjusted rand index score for fluid communities clusterings against IAC."""
import main_pipelines as mp

def main():
    mp.IDENTIFIER_STRING = "dblp"
    mp.K = 2
    mp.ALPHA_VALUES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    mp.mean_ari()
    return

if __name__ == '__main__':
    main()
