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
	@staticmethod
	def resetLiteralColor():
		print(PrintColor.RESET)
	@staticmethod
	def instant(message:str,color:PrintColor)->str:
		Printer.resetLiteralColor()
		return f"{color}{message}{PrintColor.RESET}"
	@staticmethod
	def warn(message:str)->str:
		return Printer.instant(message,PrintColor.YELLOW)
	@staticmethod
	def info(message:str)->str:
		return Printer.instant(message,PrintColor.CYAN)
	@staticmethod
	def err(message:str)->str:
		return Printer.instant(message,PrintColor.RED)
	@staticmethod
	def tensorInfoPrint(tensor:torch.Tensor,output_values: bool=True):
		return "None" if tensor == None else f"shape({tensor.shape if isinstance(tensor,torch.Tensor) else [t.shape for t in tensor]})"+ (f" = {tensor}" if output_values else "")
	@staticmethod
	def tensorPrint(tensor:torch.Tensor,output_values: bool=True):
		frame = inspect.currentframe()
		names = {id(v): k for k, v in frame.f_back.f_locals.items()}
		return 	f"{names[id(tensor)]}:{Printer.tensorInfoPrint(tensor,output_values)}"
	@staticmethod
	def tensorDictPrint(tensor_dict: dict,base_output_values: bool=True,output_values: dict= None,indent: str="  ",newlines: bool= True):
		if output_values is None:
			output_values = dict()
		for k in tensor_dict.keys():
			if k not in output_values:
				output_values[k] = base_output_values
		return indent.join(["{\n"]+[f"{k} : {Printer.tensorInfoPrint(v,output_values[k])}"+Printer.ifstr(newlines,'\n') for k,v in tensor_dict.items()]) + "}"
	@staticmethod
	def anotateLine(extrainfo: str='',print_code: bool=False):
		frame = inspect.currentframe().f_back
		return f"File \"{frame.f_code.co_filename}\", line {frame.f_lineno}, in {frame.f_code.co_name}"+Printer.ifstr(extrainfo!='',extrainfo)+Printer.ifstr(print_code,"\n"+f"{frame.f_code}")
	@staticmethod
	def ifstr(conditional:bool,str:str):
		return str if conditional else ""
  

