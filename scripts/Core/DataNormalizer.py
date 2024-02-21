# Agentがあるディレクトリにnorm_config.ymlと一緒に置いておく

import math
import numpy as np
import yaml
import os
from importlib import import_module
import json
import re
import sys
import io
from PIL import Image

from torch import le
from . import *
from ..Helper.Printer import PrintColor, Printer
from ..Helper.DictExtension import DictExtension


re_comp = re.compile(r"\D")

"""
init内で
self.normalizer = DataNormalizer(modelConfig,instanceConfig)
を呼び出して、makeObs内で
self.normalizer.normalize("Observation",str(parent.observables))
を呼び出して取得できる
"""
class DataNormalizer:
    stdparam:dict = None
    def __init__(self,modelConfig,instanceConfig):
        self.load_std_params(modelConfig,instanceConfig)
        
    @classmethod
    def load_std_params(cls,modelConfig,instanceConfig):
        stdparam_dict:dict = NORM_CONFIG['stdparam']
        stdparam_class_dict:dict = stdparam_dict['class']
        #classes = []
        stdParamsPath = os.path.join(os.path.dirname(__file__),CONFIGPATH,getStdParamsPath())
        if cls.stdparam is None and os.path.exists(stdParamsPath):
            with open(stdParamsPath, 'r') as jsn:
                cls.stdparam:dict = json.load(jsn)
            print("stdparam was loaded!")
        if cls.stdparam is None:
            stdclassModule = import_module("ASRCAISim1.libCore")
            cls.stdparam = dict()
            for stdclass in stdparam_class_dict.keys():
                # print("class check =",stdclass)
                stdclasskeys = stdparam_class_dict[stdclass]
                # print("class keys =",stdclasskeys)
                stdclassinstance = getattr(stdclassModule,stdclass)(modelConfig,instanceConfig)
                for stdclasskey in stdclasskeys:
                    if stdclasskey not in cls.stdparam:
                        mayvalue = getattr(stdclassinstance,stdclasskey)
                        # print(stdclasskey,"value =",mayvalue)
                        cls.stdparam[stdclasskey] = mayvalue
            # print("Fields:",cls.stdparam)
            with open(stdParamsPath, 'w') as jsn:
                json.dump(cls.stdparam,jsn, indent=2)
        # print(cls.norm_class_size)

    def normalize(self,class_name,data):
        # print("Check data",data)
        assert isinstance(data,dict) or isinstance(data,str), f"data type should be str or dict. But data type was {type(data)}"
        if isinstance(data,str):
            data:dict = json.loads(data)
        if self.stdparam is None:
            self.load_std_params()
        subn = SubNormalizer(self.stdparam)
        normalized_data = subn.normalize(class_name,data,['Observation'])
        # print("data size =",len(normalized_data))
        # 表現力が落ちてしまうが、行動が偏りすぎる場合は有効かもしれない
        # cls.dataToPngConverter = DataToPNG()
        # cnv_d = cls.dataToPngConverter.Convert(np.array(normalized_data))
        # print("converted:",cnv_d,"/ size =",cnv_d.size)
        return normalized_data



class SubNormalizer:
    def __init__(self,stdparam):
        self.stdparam:dict = stdparam
        self.storeddatas = dict()
        

    def normalize(self,class_name,data:dict,parents=[],ignores=[]):
        normlist = []
        if self.subnormalize(CLASS_NORM[class_name],data,normlist,parents,ignores):
            pass
        return normlist
    
    def subnormalize(self,top:dict,data,normlist:list,parents=[],ignores=[]):
        b_size = len(normlist)
        if 'class' in top:
            if 'nullable' in top and top['nullable']:
                # print(SubNormalizer.atStr(parents,Printer.info(f"padding nothing value")))
                normlist.extend([0]*DictExtension.SumChildValue(CLASS_SIZE_TREE,parents))
            else:
                # print(SubNormalizer.atStr(parents,f"search into {top['class']}/ignore:{ignores}"))
                normlist.extend(self.normalize(top['class'],data,parents,ignores.copy()))
        else:
            new_ignores = []
            if len(ignores) > 0:
                for igI in range(len(ignores)):
                    ignore = ignores[igI]
                    if len(ignore) > 0:
                        if ignore[0] == parents[-1]:
                            if len(ignore) == 1:
                                # print(SubNormalizer.atStr(parents,f"{ignore[0]} was ignored"))
                                return False
                            else:
                                # print(SubNormalizer.atStr(parents,f"ignore update -> {ignore[1:]}"))
                                new_ignores.append(ignore[1:])
            if 'dtype' not in top:
                for node_key,node in top.items():
                    # print(SubNormalizer.atStr(parents,f"search into {node_key}/ignore:{new_ignores}"))
                    if node_key not in data:
                        # print(SubNormalizer.atStr(parents,f"{node_key} not in {data}"))
                        assert any([ign[0] == node_key for ign in new_ignores]), self.atStr(parents,Printer.err(f"Ivalid key {node_key} : {data}"))
                        self.subnormalize(node,data,normlist,parents+[node_key],new_ignores)
                    else:
                        self.subnormalize(node,data[node_key],normlist,parents+[node_key],new_ignores)
            else:
                normtype = top['norm']
                datatype = top['dtype']
                if normtype != 'ignore':
                    if normtype == 'pass':
                        if isinstance(data,list):
                            normlist.extend(data)
                        else:
                            normlist.append(data)
                    else:
                        params:dict = top['params']
                        if normtype == 'objsnormalize':
                            assert isinstance(data,list) or isinstance(data,dict)
                            if isinstance(params,list):
                                if all([param in CLASS_NORM for param in params]):
                                    for sI,s_data in enumerate(data):
                                        normlist.extend(self.normalize(params[sI%len(params)],s_data,parents,new_ignores))
                            else:
                                paddings = getPadding(top)
                                new_ignores.extend([[parents[-1]]+ignore.split('.') for ignore in top['ignore']] if 'ignore' in top else [])
                                filteredData = dict(enumerate(data.copy())) if isinstance(data,list) else data
                                if 'filter' in top:
                                    filterData = top['filter']
                                    if 'has' in filterData:
                                        filteredData = {fk:fd for fk,fd in filteredData.items() if SubNormalizer.checkRootValue(fd,filterData['has'])}
                                        # print(SubNormalizer.atStr(parents,f"{len(data)} data filtered to {len(filteredData)}"))
                                if 'sort' in top:
                                    hasIgnore = 'ignore' in top['sort']
                                    sortkeys = [(sk,SubNormalizer.getRootValue(s_data,top['sort']['keys'])) for sk,s_data in filteredData.items() if not (hasIgnore and all(SubNormalizer.getRootValue(s_data,[igk]) == igv for igk,igv in top['sort']['ignore'].items()))]
                                    sortkeys.sort(key=lambda x: x[1],reverse=top['sort']['reverse'])
                                else:
                                    sortkeys = enumerate(filteredData.values())
                                if isinstance(data,list):
                                    count = 0
                                    # lastlen = len(normlist)
                                    for sI,s_data in sortkeys:
                                        if count < paddings:
                                            count += 1
                                            self.subnormalize(params,filteredData[sI],normlist,parents+[count-1],[[ni[0]] + [count-1] + ni[1:] for ni in new_ignores])
                                        else:
                                            break
                                    # if count < paddings:
                                    #    self.paddingCls(normlist,params['class'],len(normlist)-lastlen,paddings-len(filteredData))
                                else:
                                    count = 0
                                    for s_key,s_data in filteredData.items():
                                        if count < paddings:
                                            count += 1
                                            # normlist.extend(SubNormalizer.onehotn(len(filteredData),int(re_comp.sub("",s_key))-1))
                                            self.subnormalize(params,s_data,normlist,parents+[count-1],[[count-1] + ni[1:] for ni in new_ignores])
                                        else:
                                            break       
                        elif normtype == 'flagvaluenormalize':
                            if isinstance(data,list):
                                if data[0] in params:
                                    addonehot = SubNormalizer.onehot(list(params.keys()),str(data[0]))
                                    normlist.extend(addonehot+[0]*(DataNormalizer.onehotSize(len(params.keys()))-len(addonehot)))
                                self.subnormalize(params[data[0]],data[1],normlist,parents+[data[0]],new_ignores)
                            elif data in params:
                                normlist.extend(SubNormalizer.onehot(list(params.keys()),str(data)))
                                self.subnormalize(params[data],data,normlist,parents+[data],new_ignores)
                        elif normtype == 'storedata':
                            # print(f"\t\tstore data[{params}]:",data)
                            self.storeddatas[params] = data.copy()
                        else:
                            b_size = len(normlist)
                            if normtype == 'value':
                                normlist.append(float(params))
                            elif normtype == 'mapping':
                                normlist.append(float(params[data]))
                            elif normtype == 'onehot':
                                normlist.extend(SubNormalizer.onehot(params,str(data)))
                            elif normtype == 'max':
                                if datatype[:-1] == 'vector':
                                    dim = int(datatype[-1])
                                    copiedData = list(data)
                                    if isinstance(params,dict):
                                        for key,value in params.items():
                                            if key == 'x' and dim > 0:
                                                copiedData[0] /= self.getValue(value)
                                            elif key == 'y' and dim > 1:
                                                copiedData[1] /= self.getValue(value)
                                            elif key == 'z' and dim > 2:
                                                copiedData[2] /= self.getValue(value)
                                    else:
                                        copiedData = [d/params for d in copiedData]
                                    normlist.extend(np.float32(copiedData))
                                else:
                                    normlist.append(np.float32(data)/self.getValue(params))
                            # if len(normlist) - b_size != norm_size:
                    norm_size = DictExtension.Search(CLASS_SIZE_TREE,parents)
                    p_total = DictExtension.SumChildValue(norm_size)
                    if p_total != len(normlist) - b_size:
                        # print(self.atStr(parents,Printer.warn(f"Please check keys, predict {norm_size} total: {p_total}/ got {len(normlist) - b_size}")))
                        assert len(normlist) - b_size <= p_total, "Its invalid"
                        normlist.extend([0]*(p_total - len(normlist) + b_size))

            # print(self.atStr(parents,f" => {len(normlist)} (+{len(normlist)-b_size})",True))
        return True
        
    def getValue(self,v):
        if isinstance(v,dict):
            if 'stdkey' in v:
                return np.float32(self.stdparam[v['stdkey']])
            elif 'storedvalue' in v:
                return SubNormalizer.getRootValue(self.storeddatas,v['storedvalue']['keys'])
        else:
            return np.float32(v)
    @staticmethod
    def atStr(dir:list,message:str,hideParents=False):
        str_dir = [str(d) for d in dir]
        p_str = ' > '.join(str_dir)
        if hideParents:
            p_str = Printer.instant(' > '.join(str_dir[:-1]),PrintColor.BLACK) + f' > {str_dir[-1]}'
        return f"{p_str}\t{message}"
    @staticmethod
    def getRootValue(top:dict,query:list):
        assert query[0] in top, f"{query[0]} is not in top ({top})"
        node = top[query[0]]
        # print("\t\tcheck node at",query[0])
        # print("\t\tdata =",node)
        for key in query[1:]:
            # print("\t\t\tkey =",key)
            node = node[key]
            # print("\t\t\tdata =",node)
        return node
    @staticmethod
    def checkRootValue(top:dict,query:list) -> bool:
        node = top
        for key in query:
            # print("\t\t\tcheck key =",key)
            if key in node:
                node = node[key]
                # print("\t\t\tdata =",node)
            else:
                return False
        return True
    @staticmethod
    def onehot(keys,key):
        return SubNormalizer.onehotn(len(keys),keys.index(key))
    
    @staticmethod
    def onehotn(patterns:int,index:int):
        onehotlist = []
        if patterns > 1:
            onehotlist = [0]*(patterns if patterns > 2 else 1)
            if patterns == 2:
                onehotlist[0] = index
            else:
                onehotlist[index] = 1
        return onehotlist



class DataToPNG:
    def __init__(self,vmin=-1,vmax=1):
        self.vmin = vmin
        self.vmax = vmax
    
    def Convert(self,data:np.ndarray,h=0,w=0):
        assert data.size > 3, f"data size should be greater than 3. However data size is {data.size}"
        b = int(math.sqrt(data.size/3))
        h = b if h == 0 else h
        w = math.ceil(data.size/(3*h)) if w == 0 else w
        data_clipped = np.clip(data,self.vmin,self.vmax)
        vmax = min(np.max(data_clipped),self.vmax)
        vmin = max(np.min(data_clipped),self.vmin)
        data_bytes = ((np.pad(data_clipped,((0,h*w*3-data.size))) - vmin)/(vmax-vmin) * 255).reshape(h,w,3).tobytes()
        # d_cnvを画像ファイルを作成せずにpng圧縮処理をし、RGB配列を得る
        png_buffer = io.BytesIO()
        Image.frombytes('RGB', (h, w), data_bytes).save(png_buffer, format='png')
        image_array = np.array(Image.open(png_buffer))
        png_buffer.close() # bufferを閉じてメモリを解放
        # RGB配列をもとのデータに戻す
        # RGB配列をもとのデータに対応する順番に並べ直して1次元化する
        data_restored = image_array.transpose(2,0,1).reshape(-1) # 順番通りに並べ直す
        data_restored = data_restored*(self.vmax-self.vmin)/255 + vmin
        # data_restored
        return data_restored[:data.size]