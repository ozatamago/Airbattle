# Copyright (c) 2021-2023 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
"""torch.nn.Moduleをjsonベースで柔軟に定義できるようにするためのユーティリティ
"""
from collections import defaultdict
import copy
import torch
import torch.nn as nn

class GenericLayers(torch.nn.Module):
    """jsonベースのconfigから様々なNNを生成するための基本クラス。
    configは以下の形式とする。
    {
        "input_shape": list, #入力テンソルの形(バッチサイズを除く)
        "layers": list[tuple[str,dict]], #各構成要素を生成するためのタプルのリスト。
                                         #タプルの第1要素はクラス名。第2要素はコンストラクタ引数のうち、入力テンソルの形に関する引数を除いたもの。
    }
    途中の各構成要素の入出力テンソルの形状は自動計算される。
    """
    def __init__(self,config):
        super().__init__()
        self.config=copy.deepcopy(config)
        self.input_shape=config['input_shape']
        self.layerCount=defaultdict(lambda:0)
        self.layers=[]
        self.forwardConfigs=[]
        dummy=torch.zeros([2,]+list(self.input_shape))
        for layerType,layerConfig in config['layers']:
            if(layerType=='Conv1d'):
                layer=torch.nn.Conv1d(in_channels=dummy.shape[1],**layerConfig)
            elif(layerType=='BatchNorm1d'):
                layer=torch.nn.BatchNorm1d(num_features=dummy.shape[1],**layerConfig)
            elif(layerType=='transpose'):
                def gen(dim0,dim1):
                    def func(x):
                        return torch.transpose(x,dim0,dim1)
                    return func
                layer=gen(layerConfig[0],layerConfig[1])
            elif(layerType=='mean'):
                def gen(**layerConfig):
                    def func(x):
                        return torch.mean(x,**layerConfig)
                    return func
                layer=gen(**layerConfig)
            elif(layerType=='Conv2d'):
                layer=torch.nn.Conv2d(in_channels=dummy.shape[1],**layerConfig)
            elif(layerType=='AdaptiveAvgPool2d'):
                layer=torch.nn.AdaptiveAvgPool2d(**layerConfig)
            elif(layerType=='AdaptiveMaxPool2d'):
                layer=torch.nn.AdaptiveMaxPool2d(**layerConfig)
            elif(layerType=='AvgPool2d'):
                layer=torch.nn.AvgPool2d(**layerConfig)
            elif(layerType=='MaxPool2d'):
                layer=torch.nn.MaxPool2d(**layerConfig)
            elif(layerType=='BatchNorm2d'):
                layer=torch.nn.BatchNorm2d(num_features=dummy.shape[1],**layerConfig)
            elif(layerType=='Flatten'):
                layer=torch.nn.Flatten(**layerConfig)
            elif(layerType=='Linear'):
                layer=torch.nn.Linear(dummy.shape[-1],**layerConfig)
            elif(layerType=='ReLU'):
                layer=torch.nn.ReLU()
            elif(layerType=='LeakyReLU'):
                layer=torch.nn.LeakyReLU()
            elif(layerType=='SiLU'):
                layer=torch.nn.SiLU()
            elif(layerType=='Tanh'):
                layer=torch.nn.Tanh()
            elif(layerType=='ResidualBlock'):
                layer=ResidualBlock(dummy.shape[1:],layerConfig)
            elif(layerType=='Concatenate'):
                layer=Concatenate(dummy.shape[1:],layerConfig)
            self.layerCount[layerType]+=1
            layerName=layerType+str(self.layerCount[layerType])
            setattr(self,layerName,layer)
            dummy=layer(dummy)
            self.layers.append(layer)
        self.output_shape=dummy.shape[1:]
    def forward(self,x):
        self.batch_size=x.shape[0]
        for layer in self.layers:
            x=layer(x)
        return x

class ResidualBlock(torch.nn.Module):
    """残差ブロック(y=x+h(x))。configは以下の形式とする。
    {
        "input_shape": list, #入力テンソルの形(バッチサイズを除く)
        "block": dict, #残差を計算するためのGenericLayersを生成するためのconfig。
    }
    """
    def __init__(self,input_shape,blockConfig):
        super().__init__()
        self.config=copy.deepcopy(blockConfig)
        self.config["input_shape"]=input_shape
        self.block=GenericLayers(self.config)
    def forward(self,x):
        return x+self.block(x)

class Concatenate(torch.nn.Module):
    """入力を複数のブロックに供給してその結果を結合するブロック。configは以下の形式とする。
    {
        "input_shape": list, #入力テンソルの形(バッチサイズを除く)
        "blocks": list[dict], #分岐・結合対象の各ブロック(GenericLayers)を生成するためのconfig。
        "dim": int, #結合対象の次元。省略時は-1(末尾)となる。

    }
    """
    def __init__(self,input_shape,config):
        super().__init__()
        self.config=copy.deepcopy(config)
        self.catDim=self.config.get("dim",-1)
        self.blocks=[]
        cnt=0
        for c in self.config["blocks"]:
            c["input_shape"]=input_shape
            block=GenericLayers(c)
            setattr(self,"Block"+str(cnt),block)
            self.blocks.append(block)
            cnt+=1
    def forward(self,x):
        return torch.cat([b(x) for b in self.blocks],dim=self.catDim)
