import torch

class TensorExtension:
    @staticmethod
    def extractSelect(dim:int,tensor:torch.Tensor,extracts):
        assert len(tensor.shape) > dim, f"can't select dim {dim}, tensor shape is {tensor.shape}"
        indices = torch.tensor([i for i in range(tensor.size(dim)) if i not in extracts])
        return torch.index_select(tensor, dim, indices) if len(indices) > 0 else None
    
    @staticmethod
    def tensor_padding(baseTensor: torch.Tensor,padTo: int,dim: int = 1, pad_value = 1e-8, dtype: torch = torch.float32):
        if baseTensor == None:
            return None
        sz = baseTensor.size(dim)
        if sz < padTo:
            pad_shape = list(baseTensor.shape)
            pad_shape[dim] = padTo-sz
            return torch.cat([baseTensor, torch.full(pad_shape,pad_value)], dim=dim).to(dtype)
        return baseTensor.to(dtype)
    @staticmethod
    def has_nan(tensor: torch.Tensor):
        return any(torch.isnan(tensor))
    