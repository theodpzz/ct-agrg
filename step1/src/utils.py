import os
import yaml
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Script with parameters from YAML file')
    parser.add_argument('--yaml_file', type=str, default='configs/default.yaml', help='YAML file containing parameters')
    return parser.parse_args()

def parse_yaml(file_path):
    with open(file_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def make_output_dir(output_dir):

    path_checkpoints = os.path.join(output_dir, "checkpoints")
    path_logs        = os.path.join(output_dir, "logs")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(path_checkpoints, exist_ok=True)
    os.makedirs(path_logs, exist_ok=True)

    return output_dir
