import argparse
import os

from resource_manager import read_config

parser = argparse.ArgumentParser()
parser.add_argument('config_path')
parser.add_argument('experiment_path')
args = parser.parse_args()

rm = read_config(args.config_path)
os.makedirs(args.experiment_path)
rm.save_config(os.path.join(args.experiment_path, 'resources.config'))
