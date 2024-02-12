import torch
import torch.nn as nn
import os
from .poca import ActorCritic

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
    actor_critic:nn.Module
    def __init__(self, observation_size, action_dim, agent_num):
        """
        ModelManagerのコンストラクタ
        :param observation_size: 観測値の次元数
        :param action_dim: 行動空間の次元数
        :param actor_num: Actorの数
        """
        # 観測値の次元数、行動空間の次元数、Actorの数を属性として保持する
        self.observation_size = observation_size
        self.action_dim = action_dim
        self.agent_num = agent_num
        # モデルのフォルダーのパスを作成する
        self.model_folder = os.path.join(os.path.dirname(__file__),f"../models/{observation_size}/{action_dim}")
        # モデルのフォルダーが存在しない場合は作成する
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)

    def save_models(self,mode:SaveMode=SaveMode.UPDATE,i=None):
        """
        CriticモデルとActorモデルをそれぞれ保存するメソッド
        :param mode: 保存モード（SaveModeの定数のいずれか）
        :param i: 保存するモデルの番号（Noneの場合は自動で決定）
        """
        # モデルのファイル名を作成する
        if mode is self.SaveMode.UPDATE:
            folderpath = self.folderpath
        else:
            i = max(self.getModelNumbers()) + (1 if mode is self.SaveMode.NEW else 0) if i is None else i
            folderpath = f"{self.model_folder}/{i}/"
        # モデルの重みを保存する
        torch.save(self.actor_critic.state_dict(), folderpath)

    def load_models(self,mode:LoadMode=LoadMode.LATEST,num=None):
        """
        CriticモデルとActorモデルの配列をそれぞれ読み込むメソッド
        :param mode: 読み込みモード（LoadModeの定数のいずれか）
        :param num: 読み込むモデルの番号（Noneの場合は自動で決定）
        """
        # モデルのファイル名を作成する
        if mode is self.LoadMode.LATEST:
            i = max(self.getModelNumbers())
            load_i = i
        else:
            load_i = max(self.getModelNumbers()) if num is None else num
            i = load_i if mode is self.LoadMode.CHOICE else (max(self.getModelNumbers()) + 1)
        self.folderpath = f"{self.model_folder}/{i}/"
        folderpath = f"{self.model_folder}/{load_i}/"
        # ActorCriticモデルの定義を作成する
        self.actor_critic = ActorCritic(self.observation_size,self.action_dim,self.agent_num,0.01)
        if mode is not self.LoadMode.NEW:
            # ActorCriticモデルの重みを読み込む
            self.actor_critic.load_state_dict(folderpath)
    
    def getModelNumbers(self):
        """
        モデルのフォルダーにあるモデルの番号のリストを返すメソッド
        :return: モデルの番号のリスト
        """
        return [int(f) for f in os.listdir(self.model_folder) if os.path.isdir(os.path.join(self.model_folder, f))]
    
