import zipfile

folder = './data/processed_data/'
p = folder + 'processed_data.zip'

with zipfile.ZipFile(p, 'r') as archive:
    archive.extractall(path=folder)
    print(f'Done')
