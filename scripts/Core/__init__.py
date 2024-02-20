from ..Helper.DictExtension import DictExtension
from functools import lru_cache
import os
import yaml

BASE_CONFIG = dict()
CONFIGPATH = '../../configs'
BASE_CONFIG_NAME = 'base_config.yml'
NORM_CONFIG = None
CLASS_SIZE_TREE = None
NORM_CLASS_SIZE = None

with open(os.path.join(os.path.dirname(__file__),CONFIGPATH,BASE_CONFIG_NAME), 'r') as yml:
    BASE_CONFIG:dict = yaml.safe_load(yml)
    print(f"{BASE_CONFIG_NAME} was loaded!")

def getBaseConfigContent(content,*valueKey):
    if len(valueKey) == 0:
        return BASE_CONFIG[content]
    else:
        return DictExtension.Search(getBaseConfigContent(content),valueKey)

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
def getActorModelName():
    return getModelValue('actorModelName')
@lru_cache(maxsize=1)
def getVModelName():
    return getModelValue('vModelName')
@lru_cache(maxsize=1)
def getStateEncoderModelName():
    return getModelValue('stateEncoderModelName')
@lru_cache(maxsize=1)
def getQModelName():
    return getModelValue('qModelName')
@lru_cache(maxsize=1)
def getNormConfigPath():
    return getConfigValue('normConfigName')
@lru_cache(maxsize=1)
def getStdParamsPath():
    return getConfigValue('stdParamsName')
@lru_cache(maxsize=2)
def getHyperParameters(model:str):
    return getModelValue('hyperParameters',model)

if NORM_CONFIG is None:
    with open(os.path.join(os.path.dirname(__file__),CONFIGPATH,getNormConfigPath()), 'r') as yml:
        NORM_CONFIG:dict = yaml.safe_load(yml)
        #print("norm_config was loaded!")

CLASS_NORM:dict = NORM_CONFIG['class']
DTYPE_SIZES = NORM_CONFIG['sizeparam']


def createSizeTree(top:dict,tree:dict = dict(),parents=[],ignores = []):
    if 'class' in top:
        if top['class'] in CLASS_SIZE_TREE:
            return CLASS_SIZE_TREE[top['class']]
        return createSizeTree(CLASS_NORM[top['class']],tree,parents,ignores)
    else:
        new_ignores = []
        if len(ignores) > 0:
            for igI in range(len(ignores)):
                ignore = ignores[igI]
                if len(ignore) > 0:
                    if ignore[0] == parents[-1]:
                        if len(ignore) == 1:
                            return None
                        else:
                            new_ignores.append(ignore[1:])
        if 'dtype' in top:
            add_node = 0
            if top['norm'] != 'ignore':
                datatype = top['dtype']
                datanorm = top['norm']
                if datatype in DTYPE_SIZES:
                    if DTYPE_SIZES[datatype] == 'relateChildren':
                        add_node = []
                        if datanorm == 'flagvaluenormalize':
                            add_node = [onehotSize(len(top['params'])),max([createSizeTree(v) for v  in top['params'].values()])]
                        else:
                            if datanorm == 'objsnormalize':
                                params = top['params']
                                paddings = getPadding(top)
                                # dynamicflag = 'dynamicflags' in top
                                new_ignores.extend([ignore.split('.') for ignore in top['ignore']] if 'ignore' in top else [])
                                if isinstance(params,list):
                                    if all([param in CLASS_NORM for param in params]):
                                        param2clsses = [{'class':param} for param in params]
                                        for param in param2clsses:
                                            addtree = createSizeTree(param)
                                            # if dynamicflag:
                                            #    addtree |= {'dflag':DataNormalizer.onehotSize(top['dynamicflags'])}
                                            add_node += [addtree] * paddings
                                elif isinstance(params,dict):
                                    addtree = createSizeTree(params)
                                    #if dynamicflag:
                                    #    addtree |= {'dflag':DataNormalizer.onehotSize(top['dynamicflags'])}
                                    add_node = [addtree] * paddings
                    else:
                        add_node = int(DTYPE_SIZES[datatype])
                elif datanorm == 'onehot':
                    add_node = onehotSize(len(top['params']))
                else:
                    add_node = int(DTYPE_SIZES['default'])
                return add_node
        else:
            for n_name,node in top.items():
                tree[n_name] = dict()
                tree[n_name] = createSizeTree(node,tree[n_name],parents+[n_name],new_ignores)
    return tree

    
def getDataSize(class_name):
    assert class_name in NORM_CLASS_SIZE, f"Unknown key {class_name}. You can select a key which {list(NORM_CLASS_SIZE.keys())}"
    return NORM_CLASS_SIZE[class_name]

@lru_cache(maxsize=1)
def getObservationSize():
    return getDataSize(getObservationClassName())

@lru_cache(maxsize=1)
def getTotalObservationSize():
    return getObservationSize()*getNumAgents() + 1
def onehotSize(length:int):
    if length == 1:
        return 0
    elif length == 2:
        return 1
    return length

def getPadding(top:dict):
    return int(top['padding']) if 'padding' in top else 1

if CLASS_SIZE_TREE is None:
    CLASS_SIZE_TREE:dict = dict()
    for class_name,class_data in CLASS_NORM.items():
        add_size_tree = dict()
        CLASS_SIZE_TREE[class_name] = createSizeTree({'class':class_name},add_size_tree)

if NORM_CLASS_SIZE is None:
    NORM_CLASS_SIZE = {k:DictExtension.SumChildValue(CLASS_SIZE_TREE,k) for k in CLASS_SIZE_TREE}
    # print("predict size = ",NORM_CLASS_SIZE)
