from ..Helper.DictSearcher import DictSearcher
from ..Helper.DataNormalizer import DataNormalizer
from functools import lru_cache
import os
import yaml

BASE_CONFIG = dict()
CONFIGPATH = '../../configs'
BASE_CONFIG_NAME = 'base_config.yml'

with open(os.path.join(os.path.dirname(__file__),CONFIGPATH,BASE_CONFIG_NAME), 'r') as yml:
    BASE_CONFIG:dict = yaml.safe_load(yml)
    print(f"{BASE_CONFIG_NAME} was loaded!")


def getBaseConfigContent(content,*valueKey):
    if len(valueKey) == 0:
        return BASE_CONFIG[content]
    else:
        return DictSearcher.Search(getBaseConfigContent(content),valueKey)

def getModelValue(*valueKey):
    return getBaseConfigContent('model',valueKey)

def getEnvValue(*valueKey):
    return getBaseConfigContent('env',valueKey)

def getConfigValue(*valueKey):
    return getBaseConfigContent('config',valueKey)

@lru_cache(maxsize=1)
def getActions():
    return getModelValue('actions')

@lru_cache(maxsize=1)
def getNumAgents():
    return getEnvValue('agents')

@lru_cache(maxsize=1)
def getObservationClassName():
    return getModelValue('observationClassName')

@lru_cache(maxsize=1)
def getNormConfigPath():
    return getConfigValue('normConfigName')
