import os
import io

class ReadHelper:
    def __init__(self):
        self.comments = []
        self.comments_category = {
            "positives": [],
            "negatives": []
        }

    def read_files_folder(self, path):
        # Walk over all files in directory and subdirectory
        for dirpath, _, filenames in os.walk(path):
            if 'neg' not in dirpath and 'pos' not in dirpath:
                continue
            for filename in filenames:
                self.read_file(dirpath + "/" + filename)
                if 'neg' in dirpath:
                    self.comments_category["negatives"].append(self.comments[-1])
                if 'pos' in dirpath:
                    self.comments_category["positives"].append(self.comments[-1])

    # Open a file and add its contents to comments list
    def read_file(self, path):
        with io.open(path, 'r') as input_file:
            read_file = input_file.read()
            self.comments.append(read_file)