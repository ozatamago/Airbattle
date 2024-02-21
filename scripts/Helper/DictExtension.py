from .Printer import Printer
import numpy as np

class DictExtension:
    """辞書操作に関するユーティリティ関数"""
    @classmethod
    def PrintStr(cls,hist:list,message:str):
        """
        履歴とメッセージを結合して出力します。

        Args:
            hist: 履歴リスト
            message: 出力メッセージ

        Returns:
            結合された文字列
        """
        return (('>'.join(hist) + " ") if len(hist) > 0 else '') + message
    @classmethod
    def check(cls,top,key,hist=[]):
        """
        キーがオブジェクト内に存在するかどうかをチェックします。

        Args:
            top: 検査対象のオブジェクト
            key: 検査対象のキー
            hist: 履歴リスト

        Raises:
            AssertionError: キーが存在しない場合
        """
        assert (isinstance(top,dict) and key in top) or (isinstance(top,list) and isinstance(key,int) and key < len(top)), cls.PrintStr(hist,Printer.err(f"invalid key: {key}"))
    @classmethod
    def SearchKeyList(cls,top:dict,keys:list,hist=[]):
        """
        キーリストを指定して、dictオブジェクト内を再帰的に検索します。

        Args:
            top: 検査対象のdictオブジェクト
            keys: 検索キーのリスト
            hist: 履歴リスト

        Returns:
            検索結果
        """
        if len(keys) == 0:
            return top
        cls.check(top,keys[0],hist)
        if len(keys) > 1:
            return cls.SearchKeyList(top[keys[0]],keys[1:],hist + [keys[0]])
        else:
            return top[keys[0]]
    @staticmethod
    def StackItems(dict:dict,add_dict:dict):
        """
        dictオブジェクトに、add_dictオブジェクトの要素を追加します。

        Args:
            dict: 対象となるdictオブジェクト
            add_dict: 追加するdictオブジェクト
        """
        for k, v in add_dict.items():
            if k in dict:
                dict[k].append(v)
            else:
                dict[k] = [v]
    @staticmethod
    def Search(top:dict,key):
        """
        キーを指定して、dictオブジェクト内を検索します。

        Args:
            top: 検査対象のdictオブジェクト
            key: 検索キー

        Returns:
            検索結果
        """
        if isinstance(key,list):
            return DictExtension.SearchKeyList(top,key)
        elif isinstance(key,tuple):
            return DictExtension.SearchKeyList(top,list(*key))
        else:
            DictExtension.check(top,key)
            return top[key]
    @staticmethod
    def SumChildValue(top:dict,key=None):
        """
        dictオブジェクト内にある子要素の値を合計します。

        Args:
            top: 対象となるdictオブジェクト
            key: 子要素のキー

        Returns:
            子要素の値の合計
        """
        if key is not None:
            parent = DictExtension.Search(top,key)
        else:
            parent = top
        allroot = parent
        # print("all root:",allroot)
        allroot = DictExtension.getAllLeaf(allroot)
        # print("flatten:",allroot)
        return sum(allroot) if isinstance(allroot,list) else allroot
            
    @staticmethod
    def getAllLeaf(top):
        """
        dictオブジェクト内にあるすべての子要素の値を取得します。

        Args:
            top: 対象となるdictオブジェクト

        Returns:
            子要素の値のリスト
        """
        values = []
        if isinstance(top,list):
            for value in top:
                # 返ってきたリーフノードの値をリストに追加する
                values.extend(DictExtension.getAllLeaf(value))
        elif isinstance(top,dict):
            # 辞書のキーと値を順に取り出す
            for key, value in top.items():
                # 返ってきたリーフノードの値をリストに追加する
                values.extend(DictExtension.getAllLeaf(value))
            # 値が辞書でなければ、リーフノードの値としてリストに追加する
        else:
            values.append(top)
        # リーフノードの値のリストを返す
        return values
    
    @staticmethod
    def reduceNone(dict:dict,set_func = None):
        dict_keys = list(dict.keys())
        for d_key in dict_keys:
            if dict[d_key] is None:
                dict.pop(d_key)
            elif set_func is not None:
                dict[d_key] = set_func(dict[d_key])
