import shutil
import torch
import torch.nn as nn
import os
from .poca import MAPOCA
from ..Helper.DictExtension import DictExtension
import random

class ModelManager:
    """
    マルチエージェント強化学習用モデル管理クラス

    MAPOCAモデルの保存・読み込み、モデル番号の取得などの機能を提供する。
    """

    class SaveMode:
        """
        モデルの保存モードを表すクラス
        """
        CHOICE = 'choice' # 既存のモデルを選択して上書きする
        UPDATE = 'update' # 現在のモデルを上書きする
        NEW = 'new' # 新しいモデルとして保存する
    class LoadMode:
        """
        モデルの読み込みモードを表すクラス
        """
        CHOICE = 'choice' # 既存のモデルを選択して読み込む
        LATEST = 'latest' # 最新のモデルを読み込む
        NEW = 'new' # 新しいモデルを作成する
        RANDOM = 'random' # ランダムなモデルを読み込む
        AGEST = 'agest' # 最も学習回数が多いモデルを読み込む
        KEY = 'key' # parents.txt のデータに従って選択して読み込む

    class CutoutMode:
        """
        モデルの保存数のカットアウト方法を表すクラス
        """
        OLD = 'old' # 最も古いモデルから優先的に削除
        YOUNG = 'young' # 最も学習回数が少ないモデルから優先的に削除
        RANDOM = 'random' # ランダムなモデルを削除
        KEY = 'key' # parent.txt に保存されているデータに従って優先的に削除
    
    def __init__(self, observation_size, action_dim,max_agents, hyperParameters: dict, new_models: int=5,cutout: int=21, cutoutMode: CutoutMode=CutoutMode.YOUNG,cutoutOption=None,cutoutReverse: bool=False,loadfromElite=False):
        """
        ModelManagerクラスの初期化

        Args:
            observation_size: 観測値の次元数
            action_dim: 行動空間の次元数
            max_agents: Actorの最大数
            hyperParameters: MAPOCAのハイパーパラメータ
            new_models: 既存モデルの数がこの数以下ならモデルを読み込まずに新規作成する
            cutout: 学習モデルの最大保存数
            cutoutMode: 学習モデルのカットアウト方法
            - CutoutMode.OLD: 最も古いモデルから優先的に削除
            - CutoutMode.YOUNG: 最も学習回数が少ないモデルから優先的に削除
            - CutoutMode.KEY: parent.txt に保存されているデータに従って優先的に削除
            - CutoutMode.RANDOM: ランダムなモデルを削除

            cutoutOption: カットアウトのオプション
            - 'CutoutMode.KEY' のときは、参照するキー[str]
            cutoutReverse: cutoutModeの優先順位を逆転するかどうか (Trueで逆転)
            loadfromElite: elitesフォルダーから読み込む
        """
        self.observation_size = observation_size
        self.action_dim = action_dim
        self.max_agents = max_agents
        self.hyperParameters = hyperParameters
        self.modelloaded = False
        self.hist = []
        self.cutout = cutout
        self.cutoutMode = cutoutMode
        self.cutoutOption = cutoutOption
        self.cutoutReverse = cutoutReverse
        self.new_models = new_models
        self.mapoca = None
        folder_name = "elites" if loadfromElite else "models"
        # モデルのフォルダーのパスを作成する
        self.model_folder = os.path.join(os.path.dirname(__file__),f"../{folder_name}/{observation_size}/{action_dim}")
        # print(f"CutoutMode: {self.cutoutMode}")

    def save_models(self,mode:SaveMode=SaveMode.UPDATE,i=None,info: dict=None):
        """
        モデルを保存する

        Args:
            mode: 保存モード
            - SaveMode.CHOICE: 既存のモデルを選択して上書きする
            - SaveMode.UPDATE: 現在のモデルを上書きする
            - SaveMode.NEW: 新しいモデルとして保存する

            i: モデル番号 (SaveMode.CHOICE または SaveMode.NEW の場合のみ必要)
        """
        if self.mapoca is not None:
            model_nums = self.getModelNumbers()
            overs = len(model_nums)-self.cutout
            if overs > 0:
                if self.cutoutMode is self.CutoutMode.YOUNG:
                    model_ages = self.getModelAges()
                    model_nums = [v for _,v in sorted(zip(model_ages,model_nums),reverse=self.cutoutReverse)]
                elif self.cutoutMode is self.CutoutMode.RANDOM:
                    model_nums = random.sample(model_nums,overs)
                elif self.cutoutMode is self.CutoutMode.KEY and self.cutoutOption is not None:
                    model_datas = [md+[model_nums[mi]] for mi,md in enumerate(self.getModelDatas(self.cutoutOption))]
                    model_nums = [v[-1] for v in sorted(model_datas,reverse=self.cutoutReverse)]
                for n in model_nums[:(overs+1)]:
                    folderpath = f"{self.model_folder}/{n}/"
                    if os.path.exists(folderpath):
                        shutil.rmtree(folderpath)
            if mode is self.SaveMode.UPDATE:
                folderpath = self.folderpath
            else:
                i = self.getModelMaxNumber() + (1 if mode is self.SaveMode.NEW else 0) if i is None else i
                folderpath = f"{self.model_folder}/{i}/"
            # モデルのフォルダーが存在しない場合は作成する
            if not os.path.exists(folderpath):
                os.makedirs(folderpath)
            # モデルの重みを保存する
            self.mapoca.save_state_dict(folderpath)
            info = dict() if (info is None or not isinstance(info,dict)) else info
            info['ModelNumber'] = i
            with open(f'{folderpath}parents.txt',"w") as f:
                f.write('\n'.join(self.hist + [DictExtension.toOneLineString(info)]))
            self.modelloaded = False
            print(f"Model saved to {folderpath}")

    def load_models(self,mode:LoadMode=LoadMode.LATEST,opt=None,strict: bool = True,force_load: bool = False):
        """
        モデルを読み込む

        Args:
            mode: 読み込みモード
            - LoadMode.CHOICE: 既存のモデルを選択して読み込む
            - LoadMode.LATEST: 最新のモデルを読み込む
            - LoadMode.NEW: 新しいモデルを作成する
            - LoadMode.RANDOM: 既存のモデルをランダムに選択して読み込む
            - LoadMode.AGEST: 学習回数が多いモデルを読み込む
            - LoadMode.KEY: parents.txt のデータに従って選択して読み込む
            
            opt: オプション
            - 'LoadMode.CHOICE' のときは、選択するモデル番号 (listでも可): None => 'LoadMode.LATEST' と同じ
            - 'LoadMode.AGEST' のときは、最も学習回数が多いものからいくつ少ないモデルまで選択範囲に入れるか: None => 最も学習回数が多いもののみ
            - 'LoadMode.KEY' のときは、[参照するキー[str],何番目まで対象にするか[int],reverse[bool][省略可(False)]]のリスト形式で指定

            strict: 重みの互換性を厳密にチェックするかどうか
            force_load: new_models を無視して読み込むかどうか (学習ではない時にはこれを True にする)
        """
        model_nums = self.getModelNumbers()
        if len(model_nums) <= self.new_models and not force_load:
            mode = self.LoadMode.NEW
        if mode is self.LoadMode.LATEST:
            i = self.getModelMaxNumber()
            load_i = i
        elif mode is self.LoadMode.RANDOM:
            i = random.choice(model_nums) if len(model_nums) > 0 else 1
            load_i = i
        elif mode is self.LoadMode.AGEST:
            i = 1
            if len(model_nums) > 0:
                ages = self.getModelAges()
                max_age = max(ages)
                n = 0 if opt == None else opt
                selects = [model_nums[ai] for ai,age in enumerate(ages) if age >= (max_age-n)]
                i = random.choice(selects)
            load_i = i
        elif mode is self.LoadMode.CHOICE:
            if isinstance(opt,list):
                assert any([(n in model_nums) for n in opt]), f"Please select from {{{model_nums}}}"
                i = random.choice([n for n in opt if (n in model_nums)])
            else:
                assert ((opt in model_nums) or opt is None), f"Please select from {{{model_nums}}}"
                i = self.getModelMaxNumber() if opt is None else opt
            load_i = i
        elif mode is self.LoadMode.KEY and isinstance(opt,list):
            params = [opt[0],opt[1] if len(opt) >= 2 else 1,opt[2] if len(opt) >= 3 else False]
            model_datas = [md+[model_nums[mi]] for mi,md in enumerate(self.getModelDatas(params[0]))]
            model_nums = [v[-1] for v in sorted(model_datas,reverse=params[2])][:params[1]]
            i = random.choice(model_nums)
            load_i = i
        else:
            load_i = self.getModelMaxNumber() if opt is None else opt
            i = self.getModelMaxNumber() + 1
        self.folderpath = f"{self.model_folder}/{i}/"
        folderpath = f"{self.model_folder}/{load_i}/"
        # MAPOCAモデルの定義を作成する
        self.mapoca = MAPOCA(self.observation_size,self.action_dim,self.max_agents,self.hyperParameters)
        print("MAPOCA created!")
        if mode is not self.LoadMode.NEW and os.path.exists(folderpath):
            # MAPOCAモデルの重みを読み込む
            print(f"{folderpath} model was loaded!")
            # readlinesで一括読み込み
            with open(f'{folderpath}parents.txt',"r") as f:
                self.hist = f.readlines()
            self.hist = [d.replace("\n", "") for d in self.hist]
            self.hist = [d for d in self.hist if len(d) > 0]
            self.mapoca.load_state_dict(folderpath,strict)
        else:
            print("Create New Model!")
        self.modelloaded = True
    
    def getModelNumbers(self,folderpath: str=None):
        """
        指定されたフォルダー内のモデル番号のリストを取得する。

        Args:
            folderpath: モデルのフォルダーパス

        Returns:
            モデル番号のリスト
        """
        if folderpath is None:
            folderpath = self.model_folder
        if not os.path.exists(folderpath):
            return []
        return [int(f) for f in os.listdir(folderpath) if os.path.isdir(os.path.join(folderpath, f)) and f != 'protects']
    
    def getModelMaxNumber(self,folderpath: str=None):
        """
        指定されたフォルダー内の最大モデル番号を取得する

        Args:
            folderpath: モデルのフォルダーパス

        Returns:
            最大モデル番号
        """
        model_nums = self.getModelNumbers(folderpath)
        return 0 if len(model_nums) == 0 else max(model_nums)
    
    def getModelAges(self,folderpath: str=None):
        return [self.getModelAge(n,folderpath) for n in self.getModelNumbers(folderpath)]
    
    def getModelAge(self,model_num: int,folderpath: str=None):
        if folderpath is None:
            folderpath = self.model_folder
        age = 0
        with open(f'{folderpath}/{model_num}/parents.txt',"r") as f:
            age = len(f.readlines())
        return age
    
    def getModelDatas(self,key: str,data_num=-1,folderpath: str=None):
        return [self.getModelData(n,key,data_num,folderpath) for n in self.getModelNumbers(folderpath)]

    def getModelData(self,model_num: int,key: str,data_num=-1,folderpath: str=None):
        if folderpath is None:
            folderpath = self.model_folder
        d_lines = []
        with open(f'{folderpath}/{model_num}/parents.txt',"r") as f:
            d_lines = f.readlines()
        if isinstance(data_num,int):
            d_lines = [d_lines[data_num]]
        elif isinstance(data_num,(list,set,tuple)):
            d_lines = [d_lines[i] for i in data_num]
        return [float(d_dict[key]) for d_dict in [DictExtension.oneLineStringToDict(d_line,default=0) for d_line in d_lines] if key in d_dict]


    
