import torch
import torch.nn as nn
import os
from .poca import MAPOCA

class ModelManager:
    """
    観測値と行動空間の次元数、Actorの数に応じてCriticモデルとActorモデルの管理を行うクラス
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
    mapoca:MAPOCA
    def __init__(self, observation_size, action_dim,max_agents, lr):
        """
        ModelManagerのコンストラクタ
        :param observation_size: 観測値の次元数
        :param action_dim: 行動空間の次元数
        :param lr: 学習率
        """
        # 観測値の次元数、行動空間の次元数、Actorの数を属性として保持する
        self.observation_size = observation_size
        self.action_dim = action_dim
        self.max_agents = max_agents
        self.lr = lr
        # モデルのフォルダーのパスを作成する
        self.model_folder = os.path.join(os.path.dirname(__file__),f"../models/{observation_size}/{action_dim}")

    def save_models(self,mode:SaveMode=SaveMode.UPDATE,i=None):
        """
        CriticモデルとActorモデルをそれぞれ保存するメソッド
        :param mode: 保存モード（SaveModeの定数のいずれか）
        :param i: 保存するモデルの番号（Noneの場合は自動で決定）
        """
        # モデルのフォルダーが存在しない場合は作成する
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
        # モデルのファイル名を作成する
        if mode is self.SaveMode.UPDATE:
            folderpath = self.folderpath
        else:
            i = self.getModelMaxNumber() + (1 if mode is self.SaveMode.NEW else 0) if i is None else i
            folderpath = f"{self.model_folder}/{i}/"
        # モデルの重みを保存する
        self.mapoca.save_state_dict(folderpath)

    def load_models(self,mode:LoadMode=LoadMode.LATEST,num=None):
        """
        CriticモデルとActorモデルの配列をそれぞれ読み込むメソッド
        :param mode: 読み込みモード（LoadModeの定数のいずれか）
        :param num: 読み込むモデルの番号（Noneの場合は自動で決定）
        """
        # モデルのファイル名を作成する
        if mode is self.LoadMode.LATEST:
            i = self.getModelMaxNumber()
            load_i = i
        else:
            load_i = self.getModelMaxNumber() if num is None else num
            i = load_i if mode is self.LoadMode.CHOICE else (self.getModelMaxNumber()+ 1)
        self.folderpath = f"{self.model_folder}/{i}/"
        folderpath = f"{self.model_folder}/{load_i}/"
        # ActorCriticモデルの定義を作成する
        self.mapoca = MAPOCA(self.observation_size,self.action_dim,self.max_agents,self.lr)
        if mode is not self.LoadMode.NEW:
            # ActorCriticモデルの重みを読み込む
            self.mapoca.load_state_dict(folderpath)
    
    def getModelNumbers(self):
        """
        モデルのフォルダーにあるモデルの番号のリストを返すメソッド
        :return: モデルの番号のリスト
        """
        if not os.path.exists(self.model_folder):
            return []
        return [int(f) for f in os.listdir(self.model_folder) if os.path.isdir(os.path.join(self.model_folder, f))]
    
    def getModelMaxNumber(self):
        """
        モデルのフォルダーにあるモデルの番号の最大値を返すメソッド(何もないときは0を返す)
        :return: モデルの番号
        """
        model_nums = self.getModelNumbers()
        return 0 if len(model_nums) == 0 else max(model_nums)
    

    
