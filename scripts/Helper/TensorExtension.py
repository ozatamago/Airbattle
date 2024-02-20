import torch

class TensorExtension:
    @staticmethod
    def extractSelect(dim:int,tensor:torch.Tensor,extracts):
        assert len(tensor.shape) > dim, f"can't select dim {dim}, tensor shape is {tensor.shape}"
        indices = torch.tensor([i for i in range(tensor.size(dim)) if i not in extracts])
        return torch.index_select(tensor, dim, indices) if len(indices) > 0 else None
    
    @staticmethod
    def tensor_padding(baseTensor: torch.Tensor,padTo: int,dim: int,pad_right: bool=True,pad_add: bool=False, pad_value = 1e-8, dtype: torch = torch.float32):
        if baseTensor == None:
            return None
        delta = padTo
        if not pad_add:
            delta = padTo - baseTensor.size(dim)
        if delta > 0:
            pad_shape = list(baseTensor.shape)
            pad_shape[dim] = delta
            return torch.cat(([baseTensor, torch.full(pad_shape,pad_value)] if pad_right else [torch.full(pad_shape,pad_value),baseTensor]), dim=dim).to(dtype)
        return baseTensor.to(dtype)
    @staticmethod
    def has_nan(tensor: torch.Tensor):
        return any(torch.isnan(tensor))
    