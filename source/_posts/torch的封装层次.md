---
title: torch的封装层次
categories:
  - 技术实践
tags:
  - 转载
  - pytorch
date: 2025-07-12 16:21:40
---
> Simon: 作为初学者,这篇文章对我非常有帮助。如果直接看pytorch的组织结构[github](https://github.com/pytorch/pytorch)，总是一头雾水，这篇文章提供了一个帮助，去了解pytorch是怎样一步步组织起来的。

> 0. cuda算子，一般是由核函数组成的.cu和.cpp文件
> 1. cuda封装，提供参数，调用算子，不进行存储
> 2. autograd算子，存储求导所需要的临时变量
> 3. function封装，在上一层的基础上做一些健壮性工作
> 4. Module封装，把函数封装成类，为的是实现永久性存储

torch中同一个功能，不同层级的封装有什么用？我应该用什么层级的封装？
-------------------------------------------------------------------

1:首先我们说在torch中你能看到的最基础封装是cuda封装，或者叫[cuda算子](https://zhida.zhihu.com/search?content_id=691781227&content_type=Answer&match_order=1&q=cuda%E7%AE%97%E5%AD%90&zhida_source=entity)，我们算它是1级封装。（方便起见我们这里忽略triton等其他方式实现的算子）

这个封装层级下你可能会看到这样的调用方式：

```text
GEMM_cuda.fwd(mat1,mat2)
GEMM_cuda.bwd(grad_o,mat1,mat2)
```

这种封装是python封装的最底层，它是在调用更底层的c++算子，实现矩阵运算的正向传播和反向传播。

c++这个层级的算子只负责计算，计算之后相应的内存空间就销毁，不会存储任何东西。

但我们知道torch是支持自动求导的，自动求导是依据链式法则实现的。一个简单的乘法：y=wx，计算w的导数：dy/dw = x，你会发现w的导数就是x，那么在我们计算w的导数时，就需要知道x，而x是正向传播时传递过来的，因此我们需要在正向传播时存下这个x。上面又说了cuda算子只负责计算，不负责存储，那么我们就需要更高一级的封装，来存储这些求导所使用的临时变量。

---

2: 在cuda算子之上的2级封装是[autograd算子](https://zhida.zhihu.com/search?content_id=691781227&content_type=Answer&match_order=1&q=autograd%E7%AE%97%E5%AD%90&zhida_source=entity)，它是通过继承torch.autograd.Function来实现的。这个层级的封装就是为了存储求导所需要的临时变量。从这一个层级开始就都是python代码了。

你可能见到这种形式的autograd算子：

```text
import torch
class TensorModelParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target, label_smoothing=0.0):
        # do something
        ctx.save_for_backward(...)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        # do something
        ... = ctx.saved_tensors
        return grad_input, None, None
```

你可以看到这种算子中会存在一个save_for_backward函数，专门存储反向传播需要的临时变量。

这种算子可以通过下面这种方式调用：

```text
ce_loss = TensorModelParallelCrossEntropy.apply(logits,labels)
```

你可以看到调用这种算子并不是通过使用它的forward或者backward函数，而是使用apply函数。这里torch会进行一些封装，例如调用apply后，会用forward进行计算，并将backward添加到tensor的grad_fn属性计算图中，求导时自动调用。会在使用了torch.no_grad()上下文，不需要求导时，自动抛弃掉save_for_backward存储的张量。但是这种层级用的也不是太常见，首先观察forward函数的输入参数和backward的输出参数。backward函数返回的梯度数量必须和forward输入参数的数量相同，但是可以用None占位。比如target是标签，label_smoothing是超参，不可学习，不需要导数，这里就会用None占位。因此当你需要某一个功能的时候，需要严格的选择你需要的autograd算子，达到最佳的计算效率，不需要计算的东西不要算。

---

3: 接下来就是第3级function封装。function封装的作用就是增加autograd算子的灵活性和健壮性，比如做一些异常检测，默认值填补，找到合适的autograd算子分发等等，比如这样：

```text
def linear_with_grad_accumulation_and_async_allreduce(input,weight,bias,lora_a=None,lora_b=None):
  assert input.is_cuda() and weight.is_cuda()
  if lora_a is not None and lora_b is not None:
    return LoraLinearWithGradAccumulationAndAsyncCommunication.apply(input,weight,bias,lora_a,lora_b)
  else:
    return LinearWithGradAccumulationAndAsyncCommunication.apply(input,weight,bias)
```

[torch.nn](https://zhida.zhihu.com/search?content_id=691781227&content_type=Answer&match_order=1&q=torch.nn&zhida_source=entity).functional里面的函数就是这一级封装，这一级的函数对于大部分的人来说已经可以拿来用了，比如：

```text
from torch.nn.functional import linear,dropout
linear(input,weight,bias)
dropout(input,p=0.1,training=True)
```

但是这个层级的封装依旧只会存储正、反向传播的临时变量，并不会存储一些持久化存在的变量。

比如看到linear函数，它的输入有input、weight、bias，其中input是一个临时变量，你的模型输入数据了，input就有，不输入就没有，输入不同的值input也不同。但是weight和bias是模型定义的时候就存在的，与你是否正向传播无关，也不会随着你输入input的值不同而改变。看到dropout函数，丢弃率p和模型当前是处于训练状态还是推理状态，也不是一个会每次都变的值。所以我们还需要一层封装来存储这些不会临时改变的东西。

---

4:这第4级封装就是torch的Module级别封装，也就是题主题目中提到的“用类实现”。类似这个样子：

```text
class Linear(torch.nn.Module):
  
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)
```

它会帮你定义持久存储的参数weight和bias，会帮你自动初始化这些参数，比如使用kaiming初始化。在你调用这个类创建的实例时，它会调用这个类的forward函数：

```text
layer = Linear(10,5,bias=False)
x = torch.randn(2,10)
y = layer(x)
```

[Module封装](https://zhida.zhihu.com/search?content_id=691781227&content_type=Answer&match_order=1&q=Module%E5%B0%81%E8%A3%85&zhida_source=entity)和autograd封装一样，调用和定义的函数名是不同的，同样是因为torch后台帮你做了一些操作，比如判断类是否有某个属性，判断类多重继承时应该调用谁的函数，给正反向传播的输入和输出添加一些钩子函数等。

到这里题主的问题，为什么要用类，为什么不用函数就已经很明确了。不想管理持久化的变量，就用Module封装，想要手动管理，就用function封装。想要自定义正反向传播的计算方法，就去写autograd算子，想炸裂提效，做算子融合，就去写cuda或者triton算子。

> 作者：真中合欢
> 链接：https://www.zhihu.com/question/677187311/answer/3780895706
