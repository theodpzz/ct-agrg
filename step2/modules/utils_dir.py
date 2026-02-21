import argparse
import yaml

def parse_arguments():
    parser = argparse.ArgumentParser(description='Script with parameters from JSON file')
    parser.add_argument('--yaml_file', type=str, default='/configs/train/default.yaml', help='YAML file containing parameters')
    return parser.parse_args()

def parse_yaml(file_path):
    with open(file_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def dict_to_argparse(dictionary):
    parser = argparse.ArgumentParser()
    for key, value in dictionary.items():
        parser.add_argument(f"--{key}", default=value)
    return parser
