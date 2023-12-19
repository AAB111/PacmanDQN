import torch
w = torch.tensor([[1,2],[4,5]],dtype=torch.float64,requires_grad=True)
function = torch.log((w+1)).sum()
function.backward()
print(w - 10 * w.grad)