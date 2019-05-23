import os
import subprocess
import argparse
import ntpath
import shutil
from extract_embeddings import extract_data
from train_model import train_model

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--path', required=True,
                help='path to guidance file')
ap.add_argument('-i', '--ipaddress', required=True,
                help='username and ip address of remote PC')
ap.add_argument('-d', '--dataset', required=True,
                help='path to dataset')
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-f", "--force", type=bool, default=False,
                help="force delete dataset folder")
args = vars(ap.parse_args())

if args['force']:
    shutil.rmtree(args['dataset'])
    os.mkdir(args['dataset'])

# start parsing guidance file
ssh = subprocess.Popen(['ssh', args['ipaddress'], 'cat',
                        args['path']], stdout=subprocess.PIPE)
for line in ssh.stdout:
    line = line.decode('utf-8').replace('\n', '')
    line_param = line.split(',')
    if len(line_param) != 2:
        continue
    # read user and image path
    user_name = line_param[0]
    file_path = line_param[1]

    file_name = ntpath.basename(file_path) + '.jpg'

    user_folder = os.path.sep.join([args['dataset'], user_name])
    if not os.path.exists(user_folder):
        os.mkdir(user_folder)

    # starting copying files..
    print(f"[INFO] Copying {file_path}  to {user_folder} ..")
    if 'http' not in file_path:
        file_path = f'/home/ubuntu/aim-cloud/application/{file_path}'
        scp_command = 'scp ' + args['ipaddress'] + \
            ':' + file_path + ' ' + user_folder
        os.system(scp_command)
    else:
        wget_command = f'wget {file_path} -P {user_folder}'
        os.system(wget_command)

# extract data from images
print('[INFO] Start extracting data ..')
extract_data(args['confidence'])
# train models
print('[INFO] Start training models ..')
train_model()
print('[INFO] Done!')
