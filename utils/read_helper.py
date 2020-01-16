import os
import io

class ReadHelper:
    def __init__(self):
        self.comments = []

    def read_files_folder(self, path):
        for dirpath, dirnames, filenames in os.walk(self):
            if 'neg' not in dirpath and 'pos' not in dirpath:
                continue
            for filename in filenames:
                self.read_file(dirpath + "/" + filename)

    def read_file(self, path):
        with io.open(path, 'r') as input_file:
            read_file = input_file.read()
            self.comments.append(read_file)