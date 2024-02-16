import torch

class TensorExtension:
    @staticmethod
    def extractSelect(tensor:torch.Tensor,extracts:list,dim:int):
        assert len(tensor.shape) > dim, f"can't select dim {dim}, tensor shape is {tensor.shape}"
        indices = torch.tensor([i for i in range(tensor.size(dim)) if i not in extracts])
        return torch.index_select(tensor, dim, indices) if len(indices) > 0 else None
    
    @staticmethod
    def tensor_padding(baseTensor: torch.Tensor,padTo: int,dim: int = 1):
        sz = baseTensor.size(dim)
        if sz < padTo:
            pad_tensor = torch.zeros(torch.index_select(baseTensor,dim,0)).repeat(dim,padTo-sz)
            return torch.cat([baseTensor, pad_tensor], dim=dim)
        return baseTensor
    