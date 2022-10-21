import os
import io
import zipfile

__author__ = "Peter Kocsis"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = [""]
__version__ = "0.4.0"
__maintainer__ = "Peter Kocsis"
__email__ = "commonroad@lists.lrz.de"
__status__ = "Integration"

chunk_size = 5 * 1024 * 1024  # 5MB


def get_data_chunks(scenario_path):
    with open(scenario_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if len(chunk) == 0:
                return
            yield chunk


def save_chunks_to_file(chunks, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        for chunk in chunks:
            f.write(chunk)


def zip_directory(path: str, output_path: str) -> str:
    zip_file_name = os.path.join(output_path, f"{os.path.basename(path)}.zip")
    zip_file = zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            zip_file.write(file_path, os.path.basename(file_path))
    zip_file.close()
    return zip_file_name


def zip_file(path: str, output_path: str) -> str:
    zip_file_name = os.path.join(output_path, f"{os.path.basename(path)}.zip")
    zip_file = zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED)
    zip_file.write(path, output_path)
    zip_file.close()
    return zip_file_name


def unzip_file(path: str, output_path: str) -> str:
    with zipfile.ZipFile(path, 'r') as zipObj:
        zipObj.extractall(output_path)
    return os.path.basename(path)