from ASRCAISim1.addons.HandyRLUtility.model import ModelBase
from . import *
from gymnasium import spaces
from .ModelManager import ModelManager

def getBatchSize(obs,space):
    if(isinstance(space,spaces.Dict)):
        k=next(iter(space))
        return getBatchSize(obs[k],space[k])
    elif(isinstance(space,spaces.Tuple)):
        return  getBatchSize(obs[0],space[0])
    else:
        return obs.shape[0]

class Manager(ModelBase):
    def __init__(self,obs_space, ac_space, action_dist_class, model_config):
        super().__init__(obs_space, ac_space, action_dist_class, model_config)
        
        self.modelmanager = ModelManager(self.getObservationSize(),getActions(),getNumAgents(),getHyperParameters('critic')['learningRate'])
        self.modelmanager.load_models(ModelManager.LoadMode.NEW)
    
    @lru_cache(maxsize=1)
    def getObservationSize(self):
        return getDataSize(getObservationClassName())
    def getModelManager(self):
        return self.modelmanager
    def forward(self, obs, hidden=None):
        return self.getModelManager().actor_critic(obs,hidden)
    def init_hidden(self,hidden=None):
        # RNNを使用しない場合、ダミーの隠れ状態を返す
        return None
    
