from .Printer import Printer

class DictSearcher:
    @classmethod
    def PrintStr(cls,hist:list,message:str):
        return (('>'.join(hist) + " ") if len(hist) > 0 else '') + message
    @classmethod
    def check(cls,top,key,hist=[]):
        assert isinstance(top,dict) and key in top, cls.PrintStr(hist,Printer.err(f"invalid key: {key}"))
    @classmethod
    def SearchKeyList(cls,top:dict,keys:list,hist=[]):
        cls.check(top,keys[0],hist)
        if len(keys) > 1:
            return cls.SearchKeyList(top[keys[0]],keys[1:],hist + [keys[0]])
        else:
            return top[keys[0]]
    @staticmethod
    def Search(top:dict,key):
        if isinstance(key,list):
            return DictSearcher.SearchKeyList(top,key)
        elif isinstance(key,tuple):
            return DictSearcher.SearchKeyList(top,list(*key))
        else:
            DictSearcher.check(top,key)
            return top[key]
    