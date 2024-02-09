from . import *
from .ModelManager import ModelManager

class Manager:
    def __init__(self,modelConfig,instanceConfig):
        self.normalizer = DataNormalizer(modelConfig,instanceConfig)
        self.modelmanager = ModelManager(self.getObservationSize(),getActions(),getNumAgents())
    
    @lru_cache(maxsize=1)
    def getObservationSize(self):
        return self.normalizer.getDataSize(getObservationClassName())
    def getModelManager(self):
        return self.modelmanager
    
    
