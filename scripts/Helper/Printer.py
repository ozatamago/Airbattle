class PrintColor:
	BLACK          = '\033[30m'#(文字)黒
	RED            = '\033[31m'#(文字)赤
	GREEN          = '\033[32m'#(文字)緑
	YELLOW         = '\033[33m'#(文字)黄
	BLUE           = '\033[34m'#(文字)青
	MAGENTA        = '\033[35m'#(文字)マゼンタ
	CYAN           = '\033[36m'#(文字)シアン
	WHITE          = '\033[37m'#(文字)白
	COLOR_DEFAULT  = '\033[39m'#文字色をデフォルトに戻す
	BOLD           = '\033[1m'#太字
	UNDERLINE      = '\033[4m'#下線
	INVISIBLE      = '\033[08m'#不可視
	REVERCE        = '\033[07m'#文字色と背景色を反転
	BG_BLACK       = '\033[40m'#(背景)黒
	BG_RED         = '\033[41m'#(背景)赤
	BG_GREEN       = '\033[42m'#(背景)緑
	BG_YELLOW      = '\033[43m'#(背景)黄
	BG_BLUE        = '\033[44m'#(背景)青
	BG_MAGENTA     = '\033[45m'#(背景)マゼンタ
	BG_CYAN        = '\033[46m'#(背景)シアン
	BG_WHITE       = '\033[47m'#(背景)白
	BG_DEFAULT     = '\033[49m'#背景色をデフォルトに戻す
	RESET          = '\033[0m'#全てリセット

import torch
import inspect
class Printer:
	"""
    ターミナル出力用の装飾や、Tensor情報出力などを補助するユーティリティクラス
    """
	@staticmethod
	def resetLiteralColor():
		"""
        文字列装飾のリセットを行う

        Returns:
            なし
        """
		print(PrintColor.RESET)
	@staticmethod
	def instant(message:str,color:PrintColor)->str:
		"""
        文字色を変更しメッセージを装飾する

        Args:
            message: 出力する文字列
            color: 文字色指定 (PrintColor 列挙型)

        Returns:
            文字色の装飾が施された文字列
        """
		Printer.resetLiteralColor()
		return f"{color}{message}{PrintColor.RESET}"
	@staticmethod
	def warn(message:str)->str:
		"""
        警告メッセージの装飾を行う

        Args:
            message: 警告メッセージ

        Returns:
            黄色で装飾された警告メッセージ
        """
		return Printer.instant(message,PrintColor.YELLOW)
	@staticmethod
	def info(message:str)->str:
		"""
        情報メッセージの装飾を行う

        Args:
            message: 情報メッセージ

        Returns:
            シアン色で装飾された情報メッセージ
        """
		return Printer.instant(message,PrintColor.CYAN)
	@staticmethod
	def err(message:str)->str:
		"""
        エラーメッセージの装飾を行う 

        Args:
            message: エラーメッセージ

        Returns:
            赤色で装飾されたエラーメッセージ
        """
		return Printer.instant(message,PrintColor.RED)
	@staticmethod
	def tensorInfoPrint(tensor:torch.Tensor,output_values: bool=True):
		"""
        PyTorch Tensorの形状情報を出力用の文字列に整形する

        Args:
            tensor: PyTorch Tensor オブジェクト
            output_values: Tensorの値自体を含めて出力するか指定  

        Returns:
            Tensorの形状情報 (値出力指定時は値も含む) を含む文字列
        """
		return "None" if tensor == None else f"shape({tensor.shape if isinstance(tensor,torch.Tensor) else [t.shape for t in tensor]})"+ (f" = {tensor}" if output_values else "")
	@staticmethod
	def tensorPrint(tensor:torch.Tensor,output_values: bool=True):
		"""
        PyTorch Tensorの形状情報とともに、現在のコードにおける変数名を出力する

        Args:
            tensor: PyTorch Tensor オブジェクト
            output_values: Tensorの値自体を含めて出力するか指定  

        Returns:
            変数名: Tensorの形状情報 (値出力指定時は値も含む) を含む文字列
        """
		frame = inspect.currentframe()
		names = {id(v): k for k, v in frame.f_back.f_locals.items()}
		return 	f"{names[id(tensor)]}:{Printer.tensorInfoPrint(tensor,output_values)}"
	@staticmethod
	def tensorDictPrint(tensor_dict: dict,base_output_values: bool=True,output_values: dict= None,indent: str="  ",newlines: bool= True):
		"""
		Tensorの辞書を整形して出力する

		Args:
			tensor_dict: 出力対象のTensorを格納した辞書
			base_output_values: 各Tensorの値も出力するかどうかの基準値
			output_values: 辞書内の各Tensorに対して個別に値を出力するかどうか指定 (省略時はbase_output_valuesを参照）
			indent: インデント用の文字列
			newlines: 改行文字の挿入を行うか指定 
		"""
		if output_values is None:
			output_values = dict()
		for k in tensor_dict.keys():
			if k not in output_values:
				output_values[k] = base_output_values
		return indent.join(["{\n"]+[f"{k} : {Printer.tensorInfoPrint(v,output_values[k])}"+Printer.ifstr(newlines,'\n') for k,v in tensor_dict.items()]) + "}"
	@staticmethod
	def anotateLine(extrainfo: str='',print_code: bool=False):
		"""
		呼び出し元のコード位置情報を取得して文字列として出力する

		Args:
			extrainfo: 付加的な情報 (任意)
			print_code: コード自体も出力に含めるかどうか
		"""
		frame = inspect.currentframe().f_back
		return f"File \"{frame.f_code.co_filename}\", line {frame.f_lineno}, in {frame.f_code.co_name}"+Printer.ifstr(extrainfo!='',extrainfo)+Printer.ifstr(print_code,"\n"+f"{frame.f_code}")
	@staticmethod
	def ifstr(conditional:bool,str:str):
		"""
		条件が成立する場合のみ指定の文字列を返す

		Args:
			conditional: 出力条件
			str: 条件が成立する場合に出力する文字列
		"""
		return str if conditional else ""
  

