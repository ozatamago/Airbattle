# Copyright (c) 2021-2023 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#ルールベースの初期行動判断モデルのようにobservationやactionを必要としないタイプのモデルの登録方法例
import os,json,yaml
import numpy as np
from ASRCAISim1.addons.HandyRLUtility.distribution import getActionDistributionClass
from ASRCAISim1.addons.HandyRLUtility.StandaloneHandyRLPolicy import StandaloneHandyRLPolicy

#①Agentクラスオブジェクトを返す関数を定義
def getUserAgentClass(args={}):
    from .MyAgent import MyAgent
    return MyAgent


#②Agentモデル登録用にmodelConfigを表すjsonを返す関数を定義
"""
json ファイル等を用意して modelConfig を記述し、Factory にモデル登録ができるようにする。こ
のとき、perceive、control、behave をオーバーライドした場合であって、1[tick]ごとではない処理
周期としたい場合には、４.１.３項に従い modelConfig に処理周期に関する記述を追加する。
なお、modelConfigとは、Agentクラスのコンストラクタに与えられる二つのjson(dict)のうちの一つであり、設定ファイルにおいて
{
    "Factory":{
        "Agent":{
            "modelName":{
                "class":"className",
                "config":{...}
            }
        }
    }
}の"config"の部分に記載される{...}のdictが該当する。
"""    
def getUserAgentModelConfig(args={}):
    # with openをしないとファイルが開きっぱなしになる
    with open(os.path.join(os.path.dirname(__file__),"agent_config.json"),"r") as f:
        configs = json.load(f)
    return configs

#③Agentの種類(一つのAgentインスタンスで1機を操作するのか、陣営全体を操作するのか)を返す関数を定義
"""AgentがAssetとの紐付けに使用するportの名称は本来任意であるが、
　簡単のために1機を操作する場合は"0"、陣営全体を操作する場合は"0"〜"機数-1"で固定とする。
"""
def isUserAgentSingleAsset(args={}):
    #1機だけならばTrue,陣営全体ならばFalseを返すこと。
    return False

#④StandalonePolicyを返す関数を定義
def getUserPolicy(args={}):
    from .scripts.Core.Manager import Manager
    import glob
    with open(os.path.join(os.path.dirname(__file__),"configs/base_config.yml"),"r") as yml:
        model_config=yaml.safe_load(yml)
    # print(model_config)
    weightPath=None
    if(args is not None):
        weightPath=args.get("weightPath",None)
    if weightPath is None:
        cwdWeights=glob.glob(os.path.join(os.path.dirname(__file__),"*.pth"))
        weightPath=cwdWeights[0] if len(cwdWeights)>0 else None
    else:
        weightPath=os.path.join(os.path.dirname(__file__),weightPath)
    isDeterministic=True #決定論的に行動させたい場合はTrue、確率論的に行動させたい場合はFalseとする。
    return StandaloneHandyRLPolicy(Manager,model_config,weightPath,getActionDistributionClass,isDeterministic)
