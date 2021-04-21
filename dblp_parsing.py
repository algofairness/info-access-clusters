import os
import csv

DATA_FILEPATH = "dblp_data/faculty_data - faculty.csv"
PUBLICATION_ITEM_CATEGORIES = {"article", "inproceedings", "proceedings", "book", "incollection", "phdthesis",
                               "mastersthesis", "www"}
DBLP_XML_FILEPATH = "dblp_data/dblp-2021-04-01.xml"
CSV_OUTPUT_FILEPATH = "dblp_data/processed_publications.csv"


def main():
    # year_to_num = year_to_prof()
    # print(year_to_num)

    parse_xml_for_publications()
    return


def year_to_prof(printing=0):
    """Counts {year: num of people with year_of_job of that year}.
    Uses: DATA_FILEPATH.
    :return: {year: num of people with year_of_job of that year}
    """
    with open(DATA_FILEPATH, 'r') as file:
        next(file).split(",")
        year_to_num = {}
        for row in file:
            row = row.split(",")
            year = -1
            for item in row:
                try:
                    year = int(item)
                except ValueError:
                    pass

            if year not in year_to_num:
                year_to_num[year] = 0
            year_to_num[year] += 1

        total = 0
        for year in sorted(year_to_num):
            if printing:
                print("{}: {}".format(year, year_to_num[year]))
            total += year_to_num[year]

        print("year_to_prof: total = {} professors".format(total))
    return year_to_num


def parse_xml_for_publications():
    """Parses the DBLP xml file and processes each publication (for memory efficiency).
    Uses: DATA_FILEPATH, DBLP_XML_FILEPATH, PUBLICATION_ITEM_CATEGORIES, CSV_OUTPUT_FILEPATH.
    :return: None
    """
    if os.path.isfile(CSV_OUTPUT_FILEPATH):
        raise ValueError("File at CSV_OUTPUT_FILEPATH already exists: please delete the file to start parsing")

    with open(DBLP_XML_FILEPATH, 'r') as file:
        # Start with the first line of the file:
        line = next(file)
        print("Start: {}".format(line))
        # Non-publication item beginning with "<" (should be 3 words and "" or "\n"):
        non_publication_items = []

        # Search for "<[publication_item]":
        while True:
            if "<" not in line:
                try:
                    line = next(file)
                    continue
                except StopIteration:
                    print("Finished parsing the file")
                    print("non_publication_items = {}".format(non_publication_items))
                    return
            else:
                # Cases of location:
                # (i) "...<[publication_item] ...>"
                # (ii) "<[publication_item] ...>"
                index_of_arrow = line.find("<")
                # line = "<[publication_item] ...>"
                line = line[index_of_arrow:]

                # Check if it's a valid publication item:
                publication_item = line[1:].split(" ")[0]
                if publication_item not in PUBLICATION_ITEM_CATEGORIES:
                    non_publication_items.append(publication_item)
                    line = line[len(publication_item) + 1:]
                    continue

                publication = []
                # Once found and legal, add to publication all strings from "[publication_item] ..."
                # to "</publication_item>"
                closing_string = "</{}>".format(publication_item)
                while closing_string not in line:
                    publication.append(line)
                    line = next(file)
                begin_index = line.find(closing_string)
                end_index = begin_index + len(closing_string)
                publication.append(line[:end_index])

                # Process the publication:
                process_publication(publication)

                # Update line to what's right after "</publication_item>", even if it's "":
                line = line[end_index:]
    return


def process_publication(publication):
    """
    Helper function that processes the publication.
    Uses: CSV_OUTPUT_FILEPATH, PUBLICATION_ITEM_CATEGORIES.
    :param publication: a list of strings.
    :return: None
    """
    if not os.path.isfile(CSV_OUTPUT_FILEPATH):
        with open(CSV_OUTPUT_FILEPATH, 'w') as output_file:
            fieldnames = ["type", "year", "number_of_authors", "author", "title"]
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()

    with open(CSV_OUTPUT_FILEPATH, 'a') as output_file:
        user_obj_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        # Induce year:
        years = []
        for line in publication:
            while "<year>" in line:
                i_begin = line.find("<year>") + len("<year>")
                i_end = line.find("</year>")
                year = int(line[i_begin:i_end])
                line = line[i_end + len("</year>"):]
                years.append(year)
        if len(years) == 0:
            year = None
        else:
            year = years

        # Induce title:
        titles = []
        for line in publication:
            while "<title>" in line:
                i_begin = line.find("<title>") + len("<title>")
                i_end = line.find("</title>")
                title = line[i_begin:i_end]
                line = line[i_end + len("</title>"):]
                titles.append(title)
        if len(titles) == 0:
            title = None
        else:
            title = titles

        # Induce type:
        publication_type = publication[-1][2:-1]
        if publication_type not in PUBLICATION_ITEM_CATEGORIES:
            raise ValueError("Processed publication type is not in PUBLICATION_ITEM_CATEGORIES")

        # Induce the number of authors:
        num_of_auth = 0
        for line in publication:
            num_of_auth += line.count("<author")

        if num_of_auth == 0:
            author = None
        else:
            # Induce author:
            authors = []
            for line in publication:
                while "<author" in line:
                    i_begin = line.find("<author") + len("<author")
                    i_end = line.find("</author>")
                    author = line[i_begin:i_end]
                    author = author.split(">")[1]
                    authors.append(author)
                    line = line[i_end + len("</author>"):]
            author = authors
            if len(author) != num_of_auth:
                raise ValueError("len(author) = {}, while num_of_auth = {}".format(len(author), num_of_auth))

        row = [publication_type, year, num_of_auth, author, title]
        user_obj_writer.writerow(row)
    return


if __name__ == '__main__':
    main()
