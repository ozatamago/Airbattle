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
from .. import Core


re_comp = re.compile(r"\D")

"""
init内で
self.normalizer = DataNormalizer(modelConfig,instanceConfig)
を呼び出して、makeObs内で
self.normalizer.normalize("Observation",str(parent.observables))
を呼び出して取得できる
"""
class DataNormalizer:
    dtype_sizes:dict = None
    norm_config:dict = None
    stdparam:dict = None
    class_norm:dict = None
    norm_class_size:dict = None
    def __init__(self,modelConfig,instanceConfig):
        self.load_norm_config(modelConfig,instanceConfig)
        
    @classmethod
    def load_norm_config(cls,modelConfig,instanceConfig):
        if cls.norm_config is None:
            with open(os.path.join(os.path.dirname(__file__),Core.CONFIGPATH,Core.getNormConfigPath()), 'r') as yml:
                cls.norm_config:dict = yaml.safe_load(yml)
                print("norm_config was loaded!")
            stdparam_dict:dict = cls.norm_config['stdparam']
            stdparam_class_dict:dict = stdparam_dict['class']
            cls.dtype_sizes:dict = cls.norm_config['sizeparam']
            #classes = []
            stdclassModule = import_module("ASRCAISim1.libCore")
            #for name, obj in inspect.getmembers(stdclassModule):
            #    if inspect.isclass(obj):
            #        classes.append(name)
            #print("classes",classes)
            # print("Get field values...")
            if cls.stdparam is None:
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
            cls.class_norm:dict = cls.norm_config['class']
            if cls.norm_class_size is None:
                cls.norm_class_size:dict = dict()
                for class_name,class_data in cls.class_norm.items():
                    cls.norm_class_size[class_name] = cls.searchClassNode(cls.class_norm[class_name],class_name,[class_name])
        print(cls.norm_class_size)
        
    @classmethod
    def searchClassNode(cls,top:dict,tab='',parents=[],ignores = []) -> int:
        total_size = 0
        if isinstance(top, dict):
            if 'class' in top:
                if top['class'] in cls.norm_class_size:
                    node_size = cls.norm_class_size[top['class']]
                else:
                    node_size = cls.searchClassNode(cls.class_norm[top['class']],tab,parents,ignores.copy())
                total_size += node_size
            else:
                new_ignores = []
                if len(ignores) > 0:
                    for igI in range(len(ignores)):
                        ignore = ignores[igI]
                        if len(ignore) > 0:
                            if ignore[0] == parents[-1]:
                                if len(ignore) == 1:
                                    return total_size
                                else:
                                    new_ignores.append(ignore[1:])
                if 'dtype' not in top:
                    for node in top.values():
                        node_size = cls.searchClassNode(node,tab,parents,new_ignores)
                        total_size += node_size
                else:
                    if top['norm'] != 'ignore':
                        datatype = top['dtype']
                        datanorm = top['norm']
                        if datatype in cls.dtype_sizes:
                            if cls.dtype_sizes[datatype] == 'relateChildren':
                                if datanorm == 'flagvaluenormalize':
                                    total_size = len(top['params']) + 1
                                else:
                                    if datanorm == 'objsnormalize':
                                        params = top['params']
                                        paddings = int(top['padding']) if 'padding' in top else 1
                                        new_ignores.extend([[parents[-1]]+ignore.split('.') for ignore in top['ignore']] if 'ignore' in top else [])
                                        if isinstance(params,list):
                                            if all([param in cls.norm_config for param in params]):
                                                for param in params:
                                                    total_size += cls.searchClassNode(cls.class_norm[param],tab,parents,new_ignores) * paddings
                                        elif isinstance(params,dict):
                                            total_size = cls.searchClassNode(params,tab,parents,new_ignores) * paddings
                            else:
                                total_size = int(cls.dtype_sizes[datatype])
                        elif datanorm == 'onehot':
                            total_size = len(top['params'])
                        else:
                            total_size = int(cls.dtype_sizes['default'])
        else:
            total_size = 1
        return total_size

    def normalize(self,class_name,data):
        # print("Check data",data)
        assert isinstance(data,dict) or isinstance(data,str), f"data type should be str or dict. But data type was {type(data)}"
        if isinstance(data,str):
            data:dict = json.loads(data)
        if self.norm_config is None:
            self.load_norm_config()
        subn = SubNormalizer(self.norm_config,self.stdparam,self.norm_class_size)
        normalized_data = subn.normalize(class_name,data)
        print("data size =",len(normalized_data))
        # 表現力が落ちてしまうが、行動が偏りすぎる場合は有効かもしれない
        # cls.dataToPngConverter = DataToPNG()
        # cnv_d = cls.dataToPngConverter.Convert(np.array(normalized_data))
        # print("converted:",cnv_d,"/ size =",cnv_d.size)
        return normalized_data
    
    def getDataSize(self,class_name):
        assert class_name not in self.norm_class_size, f"Unknown key {class_name}. You can select a key which {list(self.norm_class_size.keys())}"
        return self.norm_class_size[class_name]


class SubNormalizer:
    def __init__(self,norm_config,stdparam,norm_class_size):
        self.norm_config = norm_config['class']
        self.stdparam:dict = stdparam
        self.storeddatas = dict()
        self.norm_class_size = norm_class_size

    def normalize(self,class_name,data:dict,parents=[],ignores=[]):
        normlist = []
        if self.subnormalize(self.norm_config[class_name],data,normlist,parents,ignores):
            # print(SubNormalizer.atStr(parents,f"{class_name} size = {len(normlist)}"))
            self.paddingCls(normlist,class_name,len(normlist))
        return normlist
    
    def subnormalize(self,top:dict,data,normlist:list,parents=[],ignores=[]):
        if 'class' in top:
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
            if isinstance(data,dict) and len(data) == 0:
                return False
            if 'dtype' not in top:
                for node_key,node in top.items():
                    # print(SubNormalizer.atStr(parents,f"search into {node_key}/ignore:{new_ignores}"))
                    if node_key not in data:
                        # print(SubNormalizer.atStr(parents,f"{node_key} not in {data}"))
                        assert any([ign[0] == node_key for ign in  new_ignores]), f"Ivalid key {node_key}"
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
                        if normtype == 'value':
                            normlist.append(params)
                        elif normtype == 'mapping':
                            normlist.append(params[data])
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
                                normlist.extend(np.double(copiedData))
                            else:
                                normlist.append(np.double(data)/self.getValue(params))
                        elif normtype == 'objsnormalize':
                            assert isinstance(data,list) or isinstance(data,dict)
                            if isinstance(params,list):
                                if all([param in self.norm_config for param in params]):
                                    for sI,s_data in enumerate(data):
                                        normlist.extend(self.normalize(params[sI%len(params)],s_data,parents,new_ignores))
                            else:
                                paddings = int(top['padding']) if 'padding' in top else 1
                                new_ignores.extend([[parents[-1]]+ignore.split('.') for ignore in top['ignore']] if 'ignore' in top else [])
                                filteredData = dict(enumerate(data.copy())) if isinstance(data,list) else data
                                if 'filter' in top:
                                    filterData = top['filter']
                                    if 'has' in filterData:
                                        filteredData = {fk:fd for fk,fd in filteredData.items() if SubNormalizer.checkRootValue(fd,filterData['has'])}
                                        # print(SubNormalizer.atStr(parents,f"{len(data)} data filtered to {len(filteredData)}"))
                                if 'sort' in top:
                                    sortkeys = [(sk,SubNormalizer.getRootValue(s_data,top['sort']['keys'])) for sk,s_data in filteredData.items()]
                                    sortkeys.sort(key=lambda x: x[1],reverse=top['sort']['reverse'])
                                else:
                                    sortkeys = enumerate(filteredData.values())
                                if isinstance(data,list):
                                    count = 0
                                    lastlen = len(normlist)
                                    for sI,s_data in sortkeys:
                                        if count < paddings:
                                            count += 1
                                            self.subnormalize(params,filteredData[sI],normlist,parents,new_ignores)
                                        else:
                                            break
                                    if len(filteredData) < paddings:
                                        self.paddingCls(normlist,params['class'],len(normlist)-lastlen,paddings-len(filteredData))
                                else:
                                    for s_key,s_data in filteredData.items():
                                        normlist.extend(SubNormalizer.onehotn(len(filteredData),int(re_comp.sub("",s_key))-1))
                                        self.subnormalize(params,s_data,normlist,parents,new_ignores)
                                
                        elif normtype == 'flagvaluenormalize':
                            if isinstance(data,list):
                                if data[0] in params:
                                    normlist.extend(SubNormalizer.onehot(list(params.keys()),str(data[0])))
                                self.subnormalize(params[data[0]],data[1],normlist,parents+[data[0]],new_ignores)
                            elif data in params:
                                normlist.extend(SubNormalizer.onehot(list(params.keys()),str(data)))
                                self.subnormalize(params[data],data,normlist,parents+[data],new_ignores)
                        elif normtype == 'storedata':
                            # print(f"\t\tstore data[{params}]:",data)
                            self.storeddatas[params] = data.copy()
        # print(SubNormalizer.atStr(parents,f"all total:{len(normlist)}"))
        return True
        
    def getValue(self,v):
        if isinstance(v,dict):
            if 'stdkey' in v:
                return np.double(self.stdparam[v['stdkey']])
            elif 'storedvalue' in v:
                return SubNormalizer.getRootValue(self.storeddatas,v['storedvalue']['keys'])
        else:
            return np.double(v)

    def paddingCls(self,src:list,class_name:str,alreadydatas:int,repeat:int=1,value=0):
        assert class_name in self.norm_class_size, f"Unknown class or class size: {class_name}"
        # print("padding ",class_name,"+",self.norm_class_size[class_name],f"* {repeat}" if repeat > 1 else "","-",alreadydatas)
        if alreadydatas < self.norm_class_size[class_name]*repeat:
            src.extend([value]*(self.norm_class_size[class_name]*repeat-alreadydatas))

    @staticmethod
    def atStr(dir:list,message:str):
        return f"{' > '.join(dir)}\t{message}"
    @staticmethod
    def getRootValue(top:dict,query:list):
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