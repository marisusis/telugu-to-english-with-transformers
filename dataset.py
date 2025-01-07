import os


class PairedTextDataset():

    def __init__(self, source_file, target_file):
        self.source_file = source_file
        self.target_file = target_file

        if not os.path.exists(source_file) or not os.path.exists(target_file):
            raise FileNotFoundError("Source or target file not found")


if __name__ == "__main__":
    pass