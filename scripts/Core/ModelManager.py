import torch
import torch.nn as nn
import os
from .poca import MAPOCA

class ModelManager:
    """
    マルチエージェント強化学習用モデル管理クラス

    MAPOCAモデルの保存・読み込み、モデル番号の取得などの機能を提供する。

    **属性**

    * observation_size: 観測値の次元数
    * action_dim: 行動空間の次元数
    * max_agents: Actorの最大数
    * lr: 学習率
    * modelloaded: モデルが読み込まれているかどうか
    * model_folder: モデルのフォルダーパス

    **メソッド**

    * save_models(mode:SaveMode=SaveMode.UPDATE,i=None): モデルを保存する
    * load_models(mode:LoadMode=LoadMode.LATEST,num=None,strict: bool = True): モデルを読み込む
    * getModelNumbers(filepath: str=None): モデル番号のリストを取得する
    * getModelMaxNumber(filepath: str=None): 最大のモデル番号を取得する
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
    
    def __init__(self, observation_size, action_dim,max_agents, lr):
        """
        ModelManagerクラスの初期化

        Args:
            observation_size: 観測値の次元数
            action_dim: 行動空間の次元数
            max_agents: Actorの最大数
            lr: 学習率
        """
        self.observation_size = observation_size
        self.action_dim = action_dim
        self.max_agents = max_agents
        self.lr = lr
        self.modelloaded = False
        # モデルのフォルダーのパスを作成する
        self.model_folder = os.path.join(os.path.dirname(__file__),f"../models/{observation_size}/{action_dim}")

    def save_models(self,mode:SaveMode=SaveMode.UPDATE,i=None):
        """
        モデルを保存する

        Args:
            mode: 保存モード
            - SaveMode.CHOICE: 既存のモデルを選択して上書きする
            - SaveMode.UPDATE: 現在のモデルを上書きする
            - SaveMode.NEW: 新しいモデルとして保存する

            i: モデル番号 (SaveMode.CHOICE または SaveMode.NEW の場合のみ必要)
        """
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

    def load_models(self,mode:LoadMode=LoadMode.LATEST,num=None,strict: bool = True):
        """
        モデルを読み込む

        Args:
            mode: 読み込みモード
            - LoadMode.CHOICE: 既存のモデルを選択して読み込む
            - LoadMode.LATEST: 最新のモデルを読み込む
            - LoadMode.NEW: 新しいモデルを作成する
            
            num: モデル番号 (LoadMode.CHOICE の場合のみ必要)
            strict: 重みの互換性を厳密にチェックするかどうか
        """
        if mode is self.LoadMode.LATEST:
            i = self.getModelMaxNumber()
            load_i = i
        else:
            load_i = self.getModelMaxNumber() if num is None else num
            i = load_i if mode is self.LoadMode.CHOICE else (self.getModelMaxNumber()+ 1)
        self.folderpath = f"{self.model_folder}/{i}/"
        folderpath = f"{self.model_folder}/{load_i}/"
        # MAPOCAモデルの定義を作成する
        self.mapoca = MAPOCA(self.observation_size,self.action_dim,self.max_agents,self.lr)
        if mode is not self.LoadMode.NEW and os.path.exists(folderpath):
            # MAPOCAモデルの重みを読み込む
            self.mapoca.load_state_dict(folderpath,strict)
        self.modelloaded = True
    
    def getModelNumbers(self,filepath: str=None):
        """
        指定されたフォルダー内のモデル番号のリストを取得する。

        Args:
            filepath: モデルのフォルダーパス

        Returns:
            モデル番号のリスト
        """
        if filepath is None:
            filepath = self.model_folder
        if not os.path.exists(filepath):
            return []
        return [int(f) for f in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, f))]
    
    def getModelMaxNumber(self,filepath: str=None):
        """
        指定されたフォルダー内の最大モデル番号を取得する

        Args:
            filepath: モデルのフォルダーパス

        Returns:
            最大モデル番号
        """
        model_nums = self.getModelNumbers(filepath)
        return 0 if len(model_nums) == 0 else max(model_nums)
    

    
