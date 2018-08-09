from aihub.core.common import settings
from aihub.core.datalake.datalake import DataLake
import shutil

class DataLakeLocal(DataLake):
    def push_file(self, local_file_name, remote_file_path):
        super().push_file(local_file_name,remote_file_path)

    def fetch_file(self, remote_path):
        super().fetch_file(remote_path)