import pathlib 
import os 
import argparse
import helper 
import boto3 
import urllib 

from typing import * 

here = pathlib.Path(__file__).parent.absolute()
data_path = os.path.join(here, '..', '..', 'data')

with open(os.path.join(here, '..', 'credentials')) as f:
    key, access = [line.rstrip() for line in f.readlines()]

s3 = boto3.resource(
    's3',
    endpoint_url="https://s3-west.nrp-nautilus.io/",
    aws_access_key_id=key,
    aws_secret_access_key=access,
)

def _download_from_key(
    key: str, 
    localpath: str,
) -> None:
    """Helper function that downloads all files recursively from the given key (folder) from the braingeneersdev S3 bucket

    :param key: S3 folder (key) to start downloading recursively from
    :type key: str
    :param localpath: Optional argument, downloads to a subfolder under the data/processed/ folder # TODO add folder generation
    :type localpath: str 
    """    

    print(f'Key is {key}')
    reduced_files = helper.list_objects(key)

    if not os.path.exists(localpath):
        print(f'Download path {localpath} doesn\'t exist, creating...')
        os.makedirs(localpath, exist_ok=True)

    for f in reduced_files:
        if not os.path.isfile(os.path.join(data_path, 'processed', f.split('/')[-1])):
            print(f'Downloading {f} from S3')
            helper.download(
                f,
                os.path.join(localpath, f.split('/')[-1]) # Just the file name in the list of objects
            )

def download_clean_from_s3(
    file: str=None,
    local_path: str=None,
) -> None:
    """Downloads the cleaned data from s3 to be used in model training

    :param file: file name to download from braingeneersdev S3 bucket, defaults to None
    :type file: str, optional
    :param local_path: path to download file to, defaults to None
    :type local_path: str, optional
    """    
    os.makedirs(os.path.join(data_path, 'processed'), exist_ok=True)
    if not file: # No single file passed, so download recursively
        print('Downloading all clean data...')
        key = os.path.join('jlehrer', 'expression_data', 'processed')
        local_path = os.path.join(data_path, 'processed')

        _download_from_key(key, local_path) 
    else:
        print(f'Downloading {file} from clean data')
        local_path = (os.path.join(data_path, 'processed', file) if not local_path else local_path)
        helper.download(
            os.path.join('jlehrer', 'expression_data', 'processed', file),
            local_path
        )

def download_interim_from_s3(
    file: str=None,
    local_path: str=None,
) -> None:
    """Downloads the interim data from S3. Interim data is in the correct structural format but has not been cleaned

    :param file: _description_, defaults to None
    :type file: str, optional
    :param local_path: _description_, defaults to None
    :type local_path: str, optional
    """
    os.makedirs(os.path.join(data_path, 'interim'), exist_ok=True)

    if not file:
        print('Downloading all interim data')
        key = os.path.join('jlehrer', 'expression_data', 'interim')
        local_path = os.path.join(data_path, 'interim')
        _download_from_key(key, local_path)
    else:
        print(f'Downloading {file} from interim data')
        local_path = (os.path.join(data_path, 'interim', file) if not local_path else local_path)
        helper.download(
            os.path.join('jlehrer', 'expression_data', 'interim', file),
            local_path,
        )
        
def download_raw_from_s3(
    file: str=None,
    local_path: str=None,
) -> None:
    """Downloads the raw expression matrices from s3


    :param file: _description_, defaults to None
    :type file: str, optional
    :param local_path: _description_, defaults to None
    :type local_path: str, optional
    """
    os.makedirs(os.path.join(data_path, 'raw'), exist_ok=True)
    if not file: 
        print('Downloading all raw data')
        key = os.path.join('jlehrer', 'expression_data', 'raw')
        local_path = os.path.join(data_path, 'raw')
        _download_from_key(key, local_path)
    else:
        print(f'Downloading {file} from raw data')
        local_path = (os.path.join(data_path, 'raw', file) if not local_path else local_path)
        helper.download(
            os.path.join('jlehrer', 'expression_data', 'raw', file), 
            local_path
        )

def upload(file_name, remote_name=None) -> None:
    """
    Uploads a file to the braingeneersdev S3 bucket
    
    Parameters:
    file_name: Local file to upload
    remote_name: Key for S3 bucket. Default is file_name
    """
    if remote_name == None:
        remote_name = file_name

    s3.Bucket('braingeneersdev').upload_file(
        Filename=file_name,
        Key=remote_name,
)

def download(remote_name, file_name=None) -> None:
    """
    Downloads a file from the braingeneersdev S3 bucket 

    Parameters:
    remote_name: S3 key to download. Must be a single file
    file_name: File name to download to. Default is remote_name
    """
    if file_name == None:
        file_name == remote_name

    s3.Bucket('braingeneersdev').download_file(
        Key=remote_name,
        Filename=file_name,
    )

def download_raw_expression_matrices(
    datasets: Dict[str, Tuple[str, str]]=None,
    upload: bool=False,
    unzip: bool=True,
    datapath: str=None
) -> None:
    """Downloads all raw datasets and label sets from cells.ucsc.edu, and then unzips them locally

    :param datasets: uses helper.DATA_FILES_AND_URLS_DICT if None. Dictionary of datasets such that each key maps to a tuple containing the expression matrix csv url in the first element,
                    and the label csv url in the second url, defaults to None
    :type datasets: Dict[str, Tuple[str, str]], optional
    :param upload: Whether or not to also upload data to the braingeneersdev S3 bucket , defaults to False
    :type upload: bool, optional
    :param unzip: Whether to also unzip expression matrix, defaults to False
    :type unzip: bool, optional
    :param datapath: Path to folder to download data to. Otherwise, defaults to data/
    :type datapath: str, optional
    """    
    # {local file name: [dataset url, labelset url]}
    datasets = (datasets if datasets is not None else helper.DATA_FILES_AND_URLS_DICT)
    data_path = (datapath if datapath is not None else os.path.join(here, '..', '..', '..', 'data'))

    for file, links in datasets.items():
        datafile_path = os.path.join(data_path, 'raw', file)

        labelfile = f'{file[:-4]}_labels.tsv'

        datalink, _ = links

        # First, make the required folders if they do not exist 
        for dir in 'raw':
            os.makedirs(os.path.join(data_path, dir), exist_ok=True)

        # Download and unzip data file if it doesn't exist 
        if not os.path.isfile(datafile_path):
            print(f'Downloading zipped data for {file}')
            urllib.request.urlretrieve(
                datalink,
                f'{datafile_path}.gz',
            )

            if unzip:
                print(f'Unzipping {file}')
                os.system(
                    f'zcat < {datafile_path}.gz > {datafile_path}'
                )

                print(f'Deleting compressed data')
                os.system(
                    f'rm -rf {datafile_path}.gz'
                )


        # If upload boolean is passed, also upload these files to the braingeneersdev s3 bucket
        if upload:
            print(f'Uploading {file} and {labelfile} to braingeneersdev S3 bucket')
            helper.upload(
                datafile_path,
                os.path.join('jlehrer', 'expression_data', 'raw', file)
            )

def download_labels(
    datasets: Dict[str, Tuple[str, str]]=None,
    upload: bool=False,
    datapath: str=None,
) -> None:
    """Downloads raw label files from given Dictionary

    :param datasets: Dictionary containing the datafile name as the key, and a tuple of the data download url and label download url as the value, defaults to None
    :type datasets: Dict[str, Tuple[str, str]], optional
    :param upload: Whether to upload data to S3, defaults to False
    :type upload: bool, optional
    :param datapath: Path to download data, defaults to None
    :type datapath: str, optional
    """    
    datasets = helper.DATA_FILES_AND_URLS_DICT
    data_path = (datapath if datapath is not None else os.path.join(here, '..', '..', '..', 'data', 'raw', 'labels'))
    
    if not os.path.isdir(data_path):
        os.makedirs(data_path, exist_ok=True)

    for labelfile, (_, labellink) in datasets.items():
        labelfile_path = os.path.join(data_path, f"{labelfile[:-4]}_labels.tsv")

        # Download label file if it doesn't exist 
        if not os.path.isfile(labelfile_path):
            print(f'Downloading label for {labelfile}')
            urllib.request.urlretrieve(
                labellink,
                labelfile_path,
            )
        else:
            print(f'{labelfile} exists, continuing...')

        if upload:
            helper.upload(
                labelfile_path,
                os.path.join('jlehrer', 'expression_data', 'raw', f'{labelfile[:-4]}_labels.tsv')
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--type',
        type=str,
        required=False,
        default='clean',
        help="Type of data to download"
    )

    parser.add_argument(
        '--key',
        required=False,
        default=None,
        type=str,
        help='If not None, only download the specific key passed in this argument from the braingeneersdev s3 bucket'
    )   

    parser.add_argument(
        '--local-name',
        required=False,
        default=None,
        help='If not None, download the key specified from the --file flag into this local filename'
    )

    args = parser.parse_args()

    type = args.type
    key = args.key 
    local = args.local_name 

    if local is not None and not key:
        parser.error('Error: If --local-name is passed in specified download, s3 key must be passed as well via --key')

    if type == 'interim':
        download_interim_from_s3(key, local)
    elif type == 'raw':
        download_raw_from_s3(key, local)
    elif type == 'processed' or type == 'clean':
        download_clean_from_s3(key, local)
    else:
        raise ValueError('Unknown type specified for data downloading.')