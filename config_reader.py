import configparser
import os


def read_config():
    file_paths = {}
    configs = {}
    current_dir = os.path.dirname(os.path.realpath(__file__))

    config = configparser.ConfigParser()
    config.read(os.path.join(current_dir, 'config.ini'))

    if 'FILES' not in config.sections():
        file_paths['detector_path'] = os.path.join(current_dir, 'output', 'embeddings.pickle')
        file_paths['embedder_path'] = os.path.join(current_dir, 'openface_nn4.small2.v1.t7')
        file_paths['recognier_path'] = os.path.join(current_dir, 'output', 'recognizer.pickle')
        file_paths['le_path'] = os.path.join(current_dir, 'output', 'le.pickle')
        return file_paths
    else:
        file_paths['detector_path'] = config['FILES']['DETECTOR']
        file_paths['embedder_path'] = config['FILES']['EMBEDDER']
        file_paths['recognier_path'] = config['FILES']['RECOGNIZER']
        file_paths['le_path'] = config['FILES']['LE']

    if 'CONFIGS' not in config.sections():
        configs['confidence'] = 0.9
    else:
        configs['confidence'] = config['CONFIGS']['CONFIDENCE']

        return file_paths, configs