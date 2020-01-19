import io

class FileHelper:
    @staticmethod
    def write_to_file(path, data):
        with io.open(path, 'w') as output_file:
            output_file.write(data)

    @staticmethod
    def read_from_file(path):
        with io.open(path, 'r') as input_file:
            read_data = input_file.read()
            return read_data