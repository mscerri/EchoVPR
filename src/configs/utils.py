import argparse
import configparser
from os.path import isfile, join

import wandb
from wandb.sdk.wandb_run import Run

from configs import ROOT_DIR


def get_config(config_file, logger):
    config = configparser.ConfigParser()
    
    logger.info(f'Root Dir {ROOT_DIR}')

    configfile = join(ROOT_DIR, config_file)
    logger.info(f'Config file {configfile}')
    assert isfile(configfile)

    config.read(configfile)

    return config

def get_config_wandb(config_file, logger, project=None, entity=None, log = True):
    config = get_config(config_file, logger)

    config_defaults = dict(config.items('main'))

    logger.info("Loaded config defaults")
    
    run = wandb.init(project=project, entity=entity, config=config_defaults)
    
    config = dict(run.config.items())

    if log:
        for c in config:
            logger.info(f'{c}: {config[c]}')
    
    return run, config

def update_config_wandb(run: Run, config, logger, log = True):
    run.config.update(config, allow_val_change=True)

    config = dict(run.config.items())

    if log:
        for c in config:
            logger.info(f'{c}: {config[c]}')
    
    return config

def update_config(config, updates):
    for k in updates:
        config[k] = updates[k]
        
    return config

def get_value_from_config(config, key, default_value):
    if key in config:
        return config[key]
    else:
        return default_value

def get_int_from_config(config, key, default_value):
    if key in config:
        return int(config[key])
    else:
        return default_value

def get_float_from_config(config, key, default_value):
    if key in config:
        return float(config[key])
    else:
        return default_value
        
def get_bool_from_config(config, key, default_value = False):
    if key in config:
        return config[key] == 'True'
    else:
        return default_value

def get_value_from_namespace(options: argparse.Namespace, key, default):
    if key in options:
        return options.__dict__[key]
    else:
        return default

def get_value_from_namespace_or_raise(options: argparse.Namespace, key):
    if key in options:
        return options.__dict__[key]
    else:
        raise Exception(f'{key} not found in options')