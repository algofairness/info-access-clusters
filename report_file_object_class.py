class ReportFileObject:

    def __init__(self, report_file_path):
        """Initializes the class instance using the report file path."""
        # Makes the path t the report file the object's attribute for easy accessing: we don't have to pass the path
        # to the print method every time we'd like to print a string.
        self.report_file_path = report_file_path
        return

    def print(self, string):
        """Writes a string both to the terminal and the report file."""
        # Prints the string in the terminal.
        print(string)
        # Opens the report file in the updating mode.
        with open(self.report_file_path, 'a') as file:
            # Writes the string to the report file.
            file.write(string)
        return
