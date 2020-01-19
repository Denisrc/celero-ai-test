import os
import io
from .file_helper import FileHelper

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
                pathname = dirpath + "/" + filename
                
                comment = FileHelper.read_from_file(pathname)

                self.comments.append(comment)
                if 'neg' in dirpath:
                    self.comments_category["negatives"].append(self.comments[-1])
                if 'pos' in dirpath:
                    self.comments_category["positives"].append(self.comments[-1])
