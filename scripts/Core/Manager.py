import sys
from ASRCAISim1.addons.HandyRLUtility.model import ModelBase
import torch
from . import *
from gymnasium import spaces
from .ModelManager import ModelManager
from ..Helper.Printer import Printer

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
        self.modelmanager = ModelManager(getObservationSize(),getActions(),getNumAgents(),getHyperParameters(),cutout=10,cutoutMode=ModelManager.CutoutMode.YOUNG,cutoutOption="RewardMean",loadfromElite=True)
    def getModelManager(self):
        return self.modelmanager
    def forward(self, obs, hidden=None):
        self.load_state_dict(None)
        return self.getModelManager().mapoca(obs,hidden)
    def init_hidden(self,hidden=None):
        # RNNを使用しない場合、ダミーの隠れ状態を返す
        return None
    def parameters(self, recurse=True):
        self.load_state_dict(None)
        # ダミー
        return self.getModelManager().mapoca.parameters(recurse)
    def updateNetworks(self,obs,rew,action_space):
        self.getModelManager().mapoca.updateNetworks(obs,rew,action_space)
        self.getModelManager().save_models(ModelManager.SaveMode.NEW,info={"TotalReward":float(torch.sum(rew).item()),"RewardMean":float(torch.mean(torch.sum(rew,dim=1)).item())})
    def load_state_dict(self, state_dict, strict: bool = True):
        if self.getModelManager().mapoca is None:
            # load_state_dictを上書きして、かわりに自作のモデル読み込み処理を行う
            self.getModelManager().load_models(ModelManager.LoadMode.CHOICE,1,strict,force_load=True)
