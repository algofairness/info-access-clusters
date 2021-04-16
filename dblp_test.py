import csv
import build_dblp_datatsets

FILE_INPUT = "dblp_data/processed_publications.csv"


def coauthorship_test():
    publications = {}
    publications["1"] = {"type": "inproceedings", "year": 2018, "number_of_authors": 5,
                         "author": ["Vikram C. M.", "S. R. Mahadeva Prasanna", "Ajish K. Abraham", "Pushpavathi M",
                                    "Girish K. S"],
                         "title": "Detection of Glottal Activity Errors in Production of Stop Consonants in Children with Cleft Lip and Palate."}
    publications["2"] = {
        "type": "article", "year": 2017, "number_of_authors": 4,
        "author": ["Ran Bi", "Jiajian Xiao", "Vaisagh Viswanathan", "Alois C. Knoll"],
        "title": "Influence of charging behaviour given charging infrastructure specification: A case study of Singapore."}

    publications["3"] = {
        "type": "inproceedings", "year": 2020, "number_of_authors": 6,
        "author": ["Chao Shi", "Gang Zhu", "Yadong Yan", "Xiangqian Chen", "Yu Wang 0083", "Kaicheng U"],
        "title": "Working Pose and Layout Optimization of surgical robot."}
    publications["4"] = {"type": "inproceedings", "year": 2011, "number_of_authors": 5, "author": ["Dongre V. B.",
                                                                                                   "R. S. Gandhi",
                                                                                                   "Anand Prakash Ruhil",
                                                                                                   "Gupta R. K.",
                                                                                                   "Singh R. K."],
                         "title": "Prediction of first lactation 305-day milk yield based on weekly test day records using artificial neural networks in Sahiwal Cattle."}
    publications["5"] = {
        "type": "article", "year": 2020, "number_of_authors": 2, "author": ["A. Amuthan", "Arulmurugan A."],
        "title": "An availability predictive trust factor-based semi-Markov mechanism for effective cluster head selection in wireless sensor networks."}
    publications["6"] = {
        "type": "article", "year": 1973, "number_of_authors": 1, "author": ["George J. Eade"],
        "title": "Military Aviation and Air Traffic Control System Planning."}
    publications["7"] = {"type": "inproceedings",
                         "year": 2018,
                         "number_of_authors": 2,
                         "author": ["Rajesh K",
                                    "Atul Negi"],
                         "title": "Heuristic Based Learning of Parameters for Dictionaries in Sparse Representations."}
    publications["8"] = {
        "type": "article", "year": 2011, "number_of_authors": 4,
        "author": ['Srinivasa K. G.', 'Anil Kumar Muppalla', 'Bharghava Varun A.', 'Amulya M.'],
        "title": "MapReduce Based Information Retrieval Algorithms for Efficient Ranking of Webpages."}
    publications["9"] = {
        "type": "inproceedings", "year": 2020, "number_of_authors": 5,
        "author": ["Rohit H. R.", "Sachin B. S.", "Aditya P.", "Bhishm Tripathi", "Premananda B. S."],
        "title": "Performance Evaluation of Various Beamforming Techniques for Phased Array Antennas."}
    publications["10"] = {
        "type": "article", "year": 2020, "number_of_authors": 2, "author": ["P. Karpagavalli", "Ramprasad A. V."],
        "title": "Automatic multiple human tracking using an adaptive hybrid GMM based detection in a crowd."}

    # Checking for correct reading and uniqueness:
    with open(FILE_INPUT, "r") as file:
        csv_reader = csv.reader(file, delimiter=',')

        rows = set()
        for row in csv_reader:
            if len(row) > 5:
                raise ValueError("More than 5 columns")

    test_results = {p: None for p in publications}
    with open(FILE_INPUT, "r") as file:
        csv_reader = csv.reader(file, delimiter=',')

        start = 1
        for row in csv_reader:
            if start:
                start -= 1
                continue
            publication_type = row[0]
            try:
                year = int(row[1][1:-1])
            except:
                year = None
            try:
                num_of_auth = int(row[2])
            except:
                raise ValueError("error")
            author = row[3][1:-1].replace("'", "").split(", ")
            title = row[4][2:-2]
            # print([publication_type, year, num_of_auth, author, title])
            for p in publications:
                if publications[p]["title"] == title and publications[p]["year"] == year and publications[p][
                    "number_of_authors"] == num_of_auth and publications[p]["author"] == author and publications[p][
                    "type"] == publication_type:
                    test_results[p] = "Correct"
    print(test_results)
    return


def compare_oct_dblp_with_2021_dblp():
    auth_2021 = set()
    with open("dblp_data/datasets_by_yoj/dblp_yoj_2021_nodelist.txt", 'r') as file:
        for line in file:
            auth_2021.add(line.split("; ")[1])
    # print(auth_2021)

    auth_oct = set()
    with open("dblp_data/drive-download-20210412T150103Z-001/dblp_nodes_oct_2020.txt", 'r') as file:
        for line in file:
            auth_oct.add(line.split("; ")[2][2:])
    # print(auth_oct)
    auth_oct.remove('lp_id')
    # print("auth_2021 - auth_oct", len(auth_2021 - auth_oct), auth_2021 - auth_oct)
    print("\nauth_oct - auth_2021", len(auth_oct - auth_2021), auth_oct - auth_2021)
    # print(auth_2021.intersection(auth_oct))

    non_unique_set = set()
    with open("dblp_data/non_unique_auth.txt", 'r') as file:
        for line in file:
            if line[-1] == "\n":
                line = line[:-1]
                non_unique_set.add(line)

    difference = auth_oct - auth_2021
    preprocessed_dif = set()
    for node in difference:
        name = build_dblp_datatsets.preprocess_id(node)
        preprocessed_dif.add(name)
    print(len(preprocessed_dif - non_unique_set), preprocessed_dif - non_unique_set)
    return


if __name__ == '__main__':
    # coauthorship_test()
    compare_oct_dblp_with_2021_dblp()
