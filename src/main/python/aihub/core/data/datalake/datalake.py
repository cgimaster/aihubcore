class DataLake():
    def push_file(self, local_file_name, remote_file_name):
        raise NotImplementedError()

    def fetch_file(self, remote_path):
        raise NotImplementedError()