from ortools.sat.python import cp_model
import torch
from typing import Dict, List, Optional
MAX_DIM = 5
MAX_SIZE = 8192
MAX_SIZE_TENSOR = 2 ** 30
MIN_SIZE_TENSOR = 2 ** 5
MAX_SIZE_PRODUCT = 2 ** 48
MAX_KERNEL_SIZE = 15
MIN_KERNEL_SIZE = 3

def infer_dtype(*input_dtypes):
    if any(d is None for d in input_dtypes):
        return None
    
    dtypes = list(input_dtypes)
    
    if len(dtypes) == 1:
        return dtypes[0]
    
    dtype_priority = {
        torch.bool: 0,
        torch.uint8: 1,
        torch.int8: 2,
        torch.int16: 3,
        torch.int32: 4,
        torch.int64: 5,
        torch.float16: 6,
        torch.bfloat16: 7,
        torch.float32: 8,
        torch.float64: 9,
        torch.complex64: 10,
        torch.complex128: 11,
    }
    
    return max(dtypes, key=lambda d: dtype_priority.get(d, -1))


def format_dtype(dtype: Optional[torch.dtype]) -> Optional[str]:
    if dtype is None:
        return None
    return str(dtype)

class ParamsDict(dict):
    """A dictionary that automatically adds decision strategies when parameters are set."""
    def __init__(self, model: cp_model.CpModel):
        super().__init__()
        self.model = model
    
    def _is_cp_var(self, value):
        """Check if value is a CP-SAT variable (IntVar or BoolVar)."""
        if value is None:
            return False
        type_str = str(type(value))
        class_name = value.__class__.__name__ if hasattr(value, '__class__') else ''
        return 'IntVar' in type_str or 'BoolVar' in type_str or class_name in ('IntVar', 'BoolVar')
    
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        # Check if value is a CP-SAT variable (IntVar or BoolVar)
        if self._is_cp_var(value):
            # Add decision strategy only for the newly added value
            self.model.add_decision_strategy(
                [value], cp_model.CHOOSE_FIRST, cp_model.SELECT_RANDOM_HALF
            )

class Edge:
    _next_id = 0
    
    def __init__(self, model: cp_model.CpModel, dtype = None, tmp = False):
        self.id = Edge._next_id
        Edge._next_id += 1
        self.model = model
        self.num = model.new_int_var(0, MAX_DIM, f'num_{self.id}')
        self.shape = [model.new_int_var(1, MAX_SIZE, f'var_{self.id}_{i}') for i in range(MAX_DIM)]
        for i in range(MAX_DIM):
            gt1 = model.new_bool_var("gt1")
            self.model.add(self.shape[i] > 1).only_enforce_if(gt1)
            self.model.add(self.shape[i] <= 1).only_enforce_if(~gt1)
            self.model.add(self.num >= MAX_DIM - i).only_enforce_if(gt1)
        
        prod = model.new_constant(1)
        for i in range(1, MAX_DIM + 1):
            geqi = model.new_bool_var("geqi")
            self.model.add(self.num >= i).only_enforce_if(geqi)
            self.model.add(self.num < i).only_enforce_if(~geqi)
            next_prod = model.new_int_var(1, MAX_SIZE_TENSOR, f'prod_{self.id}_{i}')
            new_prod = model.new_int_var(1, MAX_SIZE_TENSOR, f'new_prod_{self.id}_{i}')
            self.model.add_multiplication_equality(new_prod, [prod, self.shape[-i]])
            self.model.add(next_prod == new_prod).only_enforce_if(geqi)
            self.model.add(next_prod == prod).only_enforce_if(~geqi)
            prod = next_prod

        self.size = prod
        self.model.add(self.size <= MAX_SIZE_TENSOR)
        self.model.add(self.size >= MIN_SIZE_TENSOR)

        self.dtype = dtype
        self.tmp = tmp
        self.model.add_decision_strategy(
            self.shape + [self.num], cp_model.CHOOSE_FIRST, cp_model.SELECT_RANDOM_HALF
        )
            
    def __repr__(self):
        return f"Edge_{self.id}"

    def determine_shape(self, solver: cp_model.CpSolver):
        self.num = int(solver.Value(self.num))
        resolved_shape = [int(solver.Value(var)) for var in self.shape]
        if self.num == 0:
            self.shape = []
        else:
            self.shape = resolved_shape[-self.num:]

class BasicOp:
    def __init__(self, model: cp_model.CpModel, ins: List[Edge], dtype: torch.dtype = None):
        self.model = model
        self.ins = ins
        if ins:
            self.out = Edge(model, dtype=infer_dtype(*[i.dtype for i in ins]))
        else:
            self.out = Edge(model, dtype=dtype)
        self.params = ParamsDict(model)
        self.resolved_params: Dict[str, object] = {}

    def __repr__(self):
        tensors = ", ".join(repr(t) for t in self.ins)
        args = ", ".join([f"{k} = {v}" for k, v in self.params.items()])
        return f"{self.out} = {self.__class__.__name__}({tensors}, {args})"

    def _product_var(
        self, factors: List[cp_model.IntVar], name: str
    ) -> cp_model.IntVar:
        if not factors:
            return self.model.new_constant(1)
        if len(factors) == 1:
            return factors[0]
        bound = MAX_SIZE_PRODUCT
        prod = self.model.new_int_var(1, bound, f"{name}_{self.out.id}")
        self.model.add_multiplication_equality(prod, factors)
        return prod

    def render(self, tensor_names: Dict[int, str]) -> Optional[str]:
        raise NotImplementedError

    def render_input(self, tensor_names: Dict[int, str]) -> Optional[str]:
        return None

    def flop(self) -> cp_model.LinearExpr:
        raise NotImplementedError

class Randn(BasicOp):
    def __init__(self, model):
        super().__init__(model, [], torch.float32)
    
    def __repr__(self):
        return f"{self.out} = randn({self.out.shape})"

    def render_input(self, tensor_names: Dict[int, str]) -> str:
        dtype_name = format_dtype(self.out.dtype)
        dtype_arg = f", dtype={dtype_name}" if dtype_name else ""
        return f"    {tensor_names[self.out.id]} = torch.randn({self.out.shape}{dtype_arg})"

    def flop(self) -> cp_model.LinearExpr:
        return 0

class ElementwiseOp(BasicOp):
    torch_func: str = ""

    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins)
        assert len(ins) == 2, "ElementwiseOp must have 2 inputs"
        self.in1, self.in2 = ins

        for i in range(MAX_DIM):
            eq = self.model.new_bool_var("eq")
            in1_eq_1 = self.model.new_bool_var("in1_eq_1")
            in2_eq_1 = self.model.new_bool_var("in2_eq_1")
            self.model.add(self.in1.shape[i] == self.in2.shape[i]).only_enforce_if(eq)
            self.model.add(self.in1.shape[i] != self.in2.shape[i]).only_enforce_if(~eq)
            self.model.add(self.in1.shape[i] == 1).only_enforce_if(in1_eq_1)
            self.model.add(self.in1.shape[i] != 1).only_enforce_if(~in1_eq_1)
            self.model.add(self.in2.shape[i] == 1).only_enforce_if(in2_eq_1)
            self.model.add(self.in2.shape[i] != 1).only_enforce_if(~in2_eq_1)
            self.model.add_bool_or([eq, in1_eq_1, in2_eq_1])
            self.model.add(self.out.shape[i] == self.in1.shape[i]).only_enforce_if(eq)
            self.model.add(self.out.shape[i] == self.in1.shape[i]).only_enforce_if(in2_eq_1)
            self.model.add(self.out.shape[i] == self.in2.shape[i]).only_enforce_if(in1_eq_1)
        
        self.model.add_max_equality(self.out.num, [self.in1.num, self.in2.num])

    def render(self, tensor_names: Dict[int, str]) -> str:
        args = ", ".join(tensor_names[edge.id] for edge in self.ins)
        return f"    {tensor_names[self.out.id]} = {self.torch_func}({args})"

    def flop(self) -> cp_model.LinearExpr:
        return self.out.size

class ReduceOp(BasicOp):
    torch_func: str = ""

    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins)
        assert len(ins) == 1, "ReduceOp must have 1 input"
        self.in1 = ins[0]
        self.params["dim"] = model.new_int_var(0, MAX_DIM - 1, f'dim_{self.out.id}')
        self.params["keepdim"] = model.new_bool_var(f"keepdim_{self.out.id}")
        '''
            if keepdim:
                out[i] = in[i], if i != dim
                out[i] = 1, if i = dim
                out.num = in.num
            else:
                out[i + 1] = in[i], if dim > i
                out[i] = in[i], if dim < i
                out.num = in.num - 1

            keepdim and ~eqi  => out[i] = in[i]
            keepdim and eqi   => out[i] = 1
            ~keepdim and gti  => out[i + 1] = in[i]
            ~keepdim and ~gti and ~eqi  => out[i] = in[i]
        '''
        for i in range(MAX_DIM):
            eqi = self.model.new_bool_var("eqi")
            gti = self.model.new_bool_var("gti")
            self.model.add(self.params["dim"] + MAX_DIM - self.in1.num == i).only_enforce_if(eqi)
            self.model.add(self.params["dim"] + MAX_DIM - self.in1.num != i).only_enforce_if(~eqi)
            self.model.add(self.params["dim"] + MAX_DIM - self.in1.num > i).only_enforce_if(gti)
            self.model.add(self.params["dim"] + MAX_DIM - self.in1.num <= i).only_enforce_if(~gti)
            q1 = self.model.new_bool_var("q1")
            q2 = self.model.new_bool_var("q2")
            q3 = self.model.new_bool_var("q3")
            q4 = self.model.new_bool_var("q4")

            self.model.add_bool_or([~self.params["keepdim"], eqi, q1])
            self.model.add_bool_or([~self.params["keepdim"], ~eqi, q2])
            self.model.add_bool_or([self.params["keepdim"], ~gti, q3])
            self.model.add_bool_or([self.params["keepdim"], gti, eqi, q4])

            self.model.add(self.in1.shape[i] > 1).only_enforce_if(eqi)
            self.model.add(self.out.shape[i] == self.in1.shape[i]).only_enforce_if(q1)
            self.model.add(self.out.shape[i] == 1).only_enforce_if(q2)
            if i < MAX_DIM - 1: 
                self.model.add(self.out.shape[i + 1] == self.in1.shape[i]).only_enforce_if(q3)
            self.model.add(self.out.shape[i] == self.in1.shape[i]).only_enforce_if(q4)
            
            
        
        self.model.add(self.params["dim"] < self.in1.num)
        self.model.add(self.out.num == self.in1.num).only_enforce_if(self.params["keepdim"])
        self.model.add(self.out.num == self.in1.num - 1).only_enforce_if(~self.params["keepdim"])

    def render_kwargs(self) -> str:
        pieces: List[str] = []
        dim = self.resolved_params.get("dim")
        if dim is not None:
            pieces.append(f"dim = {dim}")
        if self.resolved_params.get("keepdim"):
            pieces.append("keepdim = True")
        return ", ".join(pieces)

    def render(self, tensor_names: Dict[int, str]) -> str:
        args = tensor_names[self.in1.id]
        line = f"    {tensor_names[self.out.id]} = {self.torch_func}({args}"
        kwargs = self.render_kwargs()
        if kwargs:
            line += f", {kwargs}"
        line += ")"
        return line

    def flop(self) -> cp_model.LinearExpr:
        return self.in1.size

class Add(ElementwiseOp):
    torch_func = "torch.add"
    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins)

class Mul(ElementwiseOp):
    torch_func = "torch.mul"
    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins)

class Sub(ElementwiseOp):
    torch_func = "torch.sub"
    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins)
    
class Div(ElementwiseOp):
    torch_func = "torch.div"
    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins)
    
class Max(ReduceOp):
    torch_func = "torch.max"
    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins)
    
    def render(self, tensor_names: Dict[int, str]) -> str:
        line = super().render(tensor_names)
        if self.resolved_params.get("dim") is not None:
            line += ".values"
        return line

class Min(ReduceOp):
    torch_func = "torch.min"
    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins)
    
    def render(self, tensor_names: Dict[int, str]) -> str:
        line = super().render(tensor_names)
        if self.resolved_params.get("dim") is not None:
            line += ".values"
        return line

class Sum(ReduceOp):
    torch_func = "torch.sum"
    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins)
    
class Mean(ReduceOp):
    torch_func = "torch.mean"
    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins)
    
class ArgMax(ReduceOp):
    torch_func = "torch.argmax"
    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins)
    
    def render(self, tensor_names: Dict[int, str]) -> str:
        line = super().render(tensor_names)
        line += ".float()"
        return line
    
class ArgMin(ReduceOp):
    torch_func = "torch.argmin"
    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins)
    
    def render(self, tensor_names: Dict[int, str]) -> str:
        line = super().render(tensor_names)
        line += ".float()"
        return line
    
class Matmul(BasicOp):
    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins)
        assert len(ins) == 2, "Matmul must have 2 inputs"
        self.in1, self.in2 = ins
    
        for i in range(MAX_DIM - 2):
            eq = self.model.new_bool_var("eq")
            in1_eq_1 = self.model.new_bool_var("in1_eq_1")
            in2_eq_1 = self.model.new_bool_var("in2_eq_1")
            self.model.add(self.in1.shape[i] == self.in2.shape[i]).only_enforce_if(eq)
            self.model.add(self.in1.shape[i] != self.in2.shape[i]).only_enforce_if(~eq)
            self.model.add(self.in1.shape[i] == 1).only_enforce_if(in1_eq_1)
            self.model.add(self.in1.shape[i] != 1).only_enforce_if(~in1_eq_1)
            self.model.add(self.in2.shape[i] == 1).only_enforce_if(in2_eq_1)
            self.model.add(self.in2.shape[i] != 1).only_enforce_if(~in2_eq_1)
            self.model.add_bool_or([eq, in1_eq_1, in2_eq_1])
            self.model.add(self.out.shape[i] == self.in1.shape[i]).only_enforce_if(eq)
            self.model.add(self.out.shape[i] == self.in1.shape[i]).only_enforce_if(in2_eq_1)
            self.model.add(self.out.shape[i] == self.in2.shape[i]).only_enforce_if(in1_eq_1)
        
        self.model.add(self.in1.shape[-1] == self.in2.shape[-2])
        self.model.add(self.out.shape[-2] == self.in1.shape[-2])
        self.model.add(self.out.shape[-1] == self.in2.shape[-1])
        self.model.add(self.in1.num >= 2)
        self.model.add(self.in2.num >= 2)
        self.model.add_max_equality(self.out.num, [self.in1.num, self.in2.num])
        
        self.product = self.model.new_int_var(1, MAX_SIZE_PRODUCT, f'product_{self.out.id}')
        self.model.add_multiplication_equality(self.product, [self.out.size, self.in1.shape[-1]])
    
    def render(self, tensor_names: Dict[int, str]) -> str:
        args = ", ".join(tensor_names[edge.id] for edge in self.ins)
        return f"    {tensor_names[self.out.id]} = torch.matmul({args})"
    
    def flop(self) -> cp_model.LinearExpr:
        return 2 * self.product

class Transpose(BasicOp):
    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins)
        assert len(ins) == 1, "Transpose requires a single input"
        self.in1 = ins[0]
        self.model.add(self.in1.num >= 2)
        for i in range(MAX_DIM - 2):
            self.model.add(self.out.shape[i] == self.in1.shape[i])
        self.model.add(self.out.shape[-2] == self.in1.shape[-1])
        self.model.add(self.out.shape[-1] == self.in1.shape[-2])
        self.model.add(self.out.num == self.in1.num)

    def render(self, tensor_names: Dict[int, str]) -> str:
        return (
            f"    {tensor_names[self.out.id]} = torch.transpose("
            f"{tensor_names[self.in1.id]}, -2, -1)"
        )
    
    def flop(self) -> cp_model.LinearExpr:
        return 0

class UnaryOp(BasicOp):
    torch_func: str = ""
    extra_args: Optional[str] = None
    flop_multiplier: int = 1

    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins)
        assert len(ins) == 1, "UnaryOp requires a single input"
        self.in1 = ins[0]
        for i in range(MAX_DIM):
            self.model.add(self.out.shape[i] == self.in1.shape[i])
        self.model.add(self.out.num == self.in1.num)

    def render(self, tensor_names: Dict[int, str]) -> str:
        extras = f", {self.extra_args}" if self.extra_args else ""
        return (
            f"    {tensor_names[self.out.id]} = {self.torch_func}("
            f"{tensor_names[self.in1.id]}{extras})"
        )

    def flop(self) -> cp_model.LinearExpr:
        return self.out.size * self.flop_multiplier

class UnaryOpWithDim(UnaryOp):
    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins)
        self.params["dim"] = model.new_int_var(0, MAX_DIM - 1, f'dim_{self.out.id}')
        self.model.add(self.params["dim"] < self.in1.num)
        # Constraint: input size on the selected dimension must be > 1
        # Map dim to the actual shape index: dim + MAX_DIM - self.in1.num
        for i in range(MAX_DIM):
            eqi = model.new_bool_var(f"eqi_{self.out.id}_{i}")
            self.model.add(self.params["dim"] + MAX_DIM - self.in1.num == i).only_enforce_if(eqi)
            self.model.add(self.params["dim"] + MAX_DIM - self.in1.num != i).only_enforce_if(~eqi)
            # If this is the selected dimension, ensure shape[i] > 1
            self.model.add(self.in1.shape[i] > 1).only_enforce_if(eqi)

    def render(self, tensor_names: Dict[int, str]) -> str:
        extras = f", {self.extra_args}" if self.extra_args else ""
        return (
            f"    {tensor_names[self.out.id]} = {self.torch_func}("
            f"{tensor_names[self.in1.id]}, dim = {self.resolved_params['dim']}{extras})"
        )

class ReLU(UnaryOp):
    torch_func = "torch.relu"
    flop_multiplier = 1

class LeakyReLU(UnaryOp):
    torch_func = "torch.nn.functional.leaky_relu"
    extra_args = "negative_slope = 0.01"
    flop_multiplier = 2

class Sigmoid(UnaryOp):
    torch_func = "torch.sigmoid"
    flop_multiplier = 4

class Tanh(UnaryOp):
    torch_func = "torch.tanh"
    flop_multiplier = 3  # tanh requires exp(2x) computation, typically ~3 FLOPs per element

class Softmax(UnaryOpWithDim):
    torch_func = "torch.softmax"
    flop_multiplier = 3
    
class LogSoftmax(UnaryOpWithDim):
    torch_func = "torch.log_softmax"
    flop_multiplier = 3
        
class Swish(UnaryOp):
    torch_func = "torch.nn.functional.silu"
    flop_multiplier = 5

class GELU(UnaryOp):
    torch_func = "torch.nn.functional.gelu"
    flop_multiplier = 8

class SELU(UnaryOp):
    torch_func = "torch.selu"
    flop_multiplier = 4

class Hardsigmoid(UnaryOp):
    torch_func = "torch.nn.functional.hardsigmoid"
    flop_multiplier = 3

class Softplus(UnaryOp):
    torch_func = "torch.nn.functional.softplus"
    flop_multiplier = 3

class Softsign(UnaryOp):
    torch_func = "torch.nn.functional.softsign"
    flop_multiplier = 3

class ELU(UnaryOp):
    torch_func = "torch.nn.functional.elu"
    extra_args = "alpha = 1.0"
    flop_multiplier = 4

class HardTanh(UnaryOp):
    torch_func = "torch.nn.functional.hardtanh"
    extra_args = "min_val = -1.0, max_val = 1.0"
    flop_multiplier = 2

class Clamp(UnaryOp):
    torch_func = "torch.clamp"
    extra_args = "min = 0.0, max = 1.0"
    flop_multiplier = 2  # Comparison operations for min and max

class BatchNorm(BasicOp):
    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins)
        assert len(ins) == 1, "BatchNorm requires one input"
        self.in1 = ins[0]
        self.model.add(self.in1.num >= 2)
        for i in range(MAX_DIM):
            eqc = self.model.new_bool_var("eqc")
            self.model.add(MAX_DIM - self.in1.num + 1 == i).only_enforce_if(eqc)
            self.model.add(MAX_DIM - self.in1.num + 1 != i).only_enforce_if(~eqc)
            self.model.add(self.out.size > self.in1.shape[i]).only_enforce_if(eqc)
            self.model.add(self.out.shape[i] == self.in1.shape[i])
        self.model.add(self.out.num == self.in1.num)

    def render(self, tensor_names: Dict[int, str]) -> str:
        channel = self.in1.shape[1]
        return (
            f"    {tensor_names[self.out.id]} = torch.nn.functional.batch_norm("
            f"{tensor_names[self.in1.id]}, torch.zeros({channel}).cuda(), "
            f"torch.ones({channel}).cuda(), None, None, training=True, momentum=0.1, eps=1e-5)"
        )

    def flop(self) -> cp_model.LinearExpr:
        return 7 * self.out.size

class LayerNorm(BasicOp):
    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins)
        assert len(ins) == 1, "LayerNorm requires one input"
        self.in1 = ins[0]
        self.params["normalized_dims"] = model.new_int_var(2, 4, f'normalized_dims_{self.out.id}')
        self.model.add(self.in1.num >= self.params["normalized_dims"])
        for i in range(MAX_DIM):
            self.model.add(self.out.shape[i] == self.in1.shape[i])
        self.model.add(self.out.num == self.in1.num)

    def render(self, tensor_names: Dict[int, str]) -> str:
        feature = self.in1.shape[-self.resolved_params["normalized_dims"]:]
        feature_str = ", ".join(str(f) for f in feature)
        return (
            f"    {tensor_names[self.out.id]} = torch.nn.functional.layer_norm("
            f"{tensor_names[self.in1.id]}, ({feature_str}), eps=1e-5)"
        )

    def flop(self) -> cp_model.LinearExpr:
        return 7 * self.out.size

class GroupNorm(BasicOp):
    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins)
        assert len(ins) == 1, "GroupNorm requires one input"
        self.in1 = ins[0]
        self.model.add(self.in1.num >= 2)
        # num_groups must be between 1 and a reasonable maximum (e.g., 32)
        # and must divide the number of channels
        self.params["num_groups"] = model.new_int_var(1, 32, f'num_groups_{self.out.id}')
        # The channel dimension is at index MAX_DIM - self.in1.num + 1
        # We need to ensure in1.shape[channel_dim] is divisible by num_groups
        channels_per_group = model.new_int_var(1, MAX_SIZE, f'channels_per_group_{self.out.id}')
        # Ensure channels are divisible by num_groups: in_channels = num_groups * channels_per_group
        for i in range(MAX_DIM):
            is_channel_dim = model.new_bool_var(f"is_channel_dim_{self.out.id}_{i}")
            self.model.add(MAX_DIM - self.in1.num + 1 == i).only_enforce_if(is_channel_dim)
            self.model.add(MAX_DIM - self.in1.num + 1 != i).only_enforce_if(~is_channel_dim)
            # For the channel dimension, ensure divisibility
            channels_times_groups = model.new_int_var(1, MAX_SIZE_PRODUCT, f'channels_times_groups_{self.out.id}_{i}')
            self.model.add_multiplication_equality(channels_times_groups, [channels_per_group, self.params["num_groups"]])
            self.model.add(self.in1.shape[i] == channels_times_groups).only_enforce_if(is_channel_dim)
            self.model.add(self.out.shape[i] == self.in1.shape[i])
        self.model.add(self.out.num == self.in1.num)

    def render(self, tensor_names: Dict[int, str]) -> str:
        num_groups = self.resolved_params["num_groups"]
        return (
            f"    {tensor_names[self.out.id]} = torch.nn.functional.group_norm("
            f"{tensor_names[self.in1.id]}, {num_groups}, eps=1e-5)"
        )

    def flop(self) -> cp_model.LinearExpr:
        return 7 * self.out.size

class InstanceNorm(BasicOp):
    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins)
        assert len(ins) == 1, "InstanceNorm requires one input"
        self.in1 = ins[0]
        self.model.add(self.in1.num >= 3)

        prod = model.new_constant(1)
        for i in range(MAX_DIM):
            self.model.add(self.out.shape[i] == self.in1.shape[i])
            in_spatial_index = self.model.new_bool_var("in_spatial_index")
            self.model.add(i >= MAX_DIM - self.in1.num + 2).only_enforce_if(in_spatial_index)
            self.model.add(i < MAX_DIM - self.in1.num + 2).only_enforce_if(~in_spatial_index)
            
            next_prod = model.new_int_var(1, MAX_SIZE_PRODUCT, f'next_prod_{self.out.id}_{i}')
            new_prod = model.new_int_var(1, MAX_SIZE_PRODUCT, f'new_prod_{self.out.id}_{i}')
            self.model.add_multiplication_equality(new_prod, [prod, self.in1.shape[i]])
            
            self.model.add(next_prod == new_prod).only_enforce_if(in_spatial_index)
            self.model.add(next_prod == prod).only_enforce_if(~in_spatial_index)
            prod = next_prod

        self.model.add(self.out.num == self.in1.num)
        self.model.add(prod > 1)

    def render(self, tensor_names: Dict[int, str]) -> str:
        return (
            f"    {tensor_names[self.out.id]} = torch.nn.functional.instance_norm("
            f"{tensor_names[self.in1.id]}, eps=1e-5)"
        )

    def flop(self) -> cp_model.LinearExpr:
        return 7 * self.out.size

class AvgPoolNd(BasicOp):
    def __init__(self, model, ins: List[Edge], nd: Optional[int] = None):
        super().__init__(model, ins)
        assert len(ins) == 1, "Pool requires one input"
        self.in1 = ins[0]
        self.params["spatial_dims"] = model.new_int_var(1, 3, f'spatial_dims_{self.out.id}')
        if nd is not None:
            self.model.add(self.params["spatial_dims"] == nd)
        self.model.add(self.in1.num >= self.params["spatial_dims"] + 1)
        self.model.add(self.in1.num <= self.params["spatial_dims"] + 2)
        self.model.add(self.out.num == self.in1.num)
        self.params["kernel_size"] = model.new_int_var(MIN_KERNEL_SIZE, MAX_KERNEL_SIZE, f'kernel_size_{self.out.id}')
        self.params["stride"] = model.new_int_var(1, MAX_KERNEL_SIZE, f'stride_{self.out.id}')
        self.params["padding"] = model.new_int_var(0, MAX_KERNEL_SIZE, f'padding_{self.out.id}')
        self.model.add(self.params["stride"] <= self.params["kernel_size"])
        self.model.add(2 * self.params["padding"] <= self.params["kernel_size"])

        for i in range(MAX_DIM):
            lt = self.model.new_bool_var("lt")
            self.model.add(i < MAX_DIM - self.params["spatial_dims"]).only_enforce_if(lt)
            self.model.add(i >= MAX_DIM - self.params["spatial_dims"]).only_enforce_if(~lt)
            self.model.add(self.out.shape[i] == self.in1.shape[i]).only_enforce_if(lt)
            '''
                d_out = (d_in - kernel_size + 2 * padding) // stride + 1
            '''
            numerator = model.new_int_var(0, 2 * MAX_SIZE, f'numerator_{self.out.id}_{i}')
            div = model.new_int_var(0, MAX_SIZE, f'div_{self.out.id}_{i}')
            
            # Need this in avgpool, but not in maxpool. 
            # Otherwise, it will raise RuntimeError: input image (T: 5 H: 5 W: 5) smaller than kernel size (kT: 7 kH: 7 kW: 7) even if padding is 1.
            self.model.add(self.in1.shape[i] >= self.params["kernel_size"]).only_enforce_if(~lt)
            
            self.model.add(numerator == self.in1.shape[i] - self.params["kernel_size"] + 2 * self.params["padding"]).only_enforce_if(~lt)
            self.model.add_division_equality(div, numerator, self.params["stride"])
            self.model.add(self.out.shape[i] == div + 1).only_enforce_if(~lt)

        # Compute kernel_volume = kernel_size^spatial_dims
        # Since spatial_dims can be 1, 2, or 3, we use conditional logic
        kernel_squared = model.new_int_var(1, MAX_KERNEL_SIZE ** 2, f'kernel_squared_{self.out.id}')
        self.model.add_multiplication_equality(kernel_squared, [self.params["kernel_size"], self.params["kernel_size"]])
        
        kernel_cubed = model.new_int_var(1, MAX_KERNEL_SIZE ** 3, f'kernel_cubed_{self.out.id}')
        self.model.add_multiplication_equality(kernel_cubed, [kernel_squared, self.params["kernel_size"]])
        
        self.kernel_volume = model.new_int_var(1, MAX_KERNEL_SIZE ** 3, f'kernel_volume_{self.out.id}')
        
        is_dim1 = model.new_bool_var(f'is_dim1_{self.out.id}')
        is_dim2 = model.new_bool_var(f'is_dim2_{self.out.id}')
        is_dim3 = model.new_bool_var(f'is_dim3_{self.out.id}')
        
        self.model.add(self.params["spatial_dims"] == 1).only_enforce_if(is_dim1)
        self.model.add(self.params["spatial_dims"] != 1).only_enforce_if(~is_dim1)
        self.model.add(self.params["spatial_dims"] == 2).only_enforce_if(is_dim2)
        self.model.add(self.params["spatial_dims"] != 2).only_enforce_if(~is_dim2)
        self.model.add(self.params["spatial_dims"] == 3).only_enforce_if(is_dim3)
        self.model.add(self.params["spatial_dims"] != 3).only_enforce_if(~is_dim3)
        
        self.model.add(self.kernel_volume == self.params["kernel_size"]).only_enforce_if(is_dim1)
        self.model.add(self.kernel_volume == kernel_squared).only_enforce_if(is_dim2)
        self.model.add(self.kernel_volume == kernel_cubed).only_enforce_if(is_dim3)
        
        # Compute kernel_volume * output_size for FLOPS
        self.kernel_volume_times_output = model.new_int_var(1, MAX_SIZE_PRODUCT, f'kernel_vol_times_out_{self.out.id}')
        self.model.add_multiplication_equality(self.kernel_volume_times_output, [self.out.size, self.kernel_volume])

    def render(self, tensor_names: Dict[int, str]) -> str:
        kernel_arg = str(self.resolved_params["kernel_size"])
        stride_arg = str(self.resolved_params["stride"])
        padding_arg = str(self.resolved_params["padding"])
        return (
            f"    {tensor_names[self.out.id]} = "
            f"torch.nn.functional.avg_pool{self.resolved_params['spatial_dims']}d("
            f"{tensor_names[self.in1.id]}, kernel_size={kernel_arg}, stride={stride_arg}, padding={padding_arg})"
        )

    def flop(self) -> cp_model.LinearExpr:
        return self.kernel_volume_times_output

class AvgPool1d(AvgPoolNd):
    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins, 1)

class AvgPool2d(AvgPoolNd):
    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins, 2)

class AvgPool3d(AvgPoolNd):
    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins, 3)

class MaxPoolNd(BasicOp):
    def __init__(self, model, ins: List[Edge], nd: Optional[int] = None):
        super().__init__(model, ins)
        assert len(ins) == 1, "Pool requires one input"
        self.in1 = ins[0]
        self.params["spatial_dims"] = model.new_int_var(1, 3, f'spatial_dims_{self.out.id}')
        if nd is not None:
            self.model.add(self.params["spatial_dims"] == nd)
        self.model.add(self.in1.num >= self.params["spatial_dims"] + 1)
        self.model.add(self.in1.num <= self.params["spatial_dims"] + 2)
        self.model.add(self.out.num == self.in1.num)
        self.params["kernel_size"] = model.new_int_var(MIN_KERNEL_SIZE, MAX_KERNEL_SIZE, f'kernel_size_{self.out.id}')
        self.params["stride"] = model.new_int_var(1, MAX_KERNEL_SIZE, f'stride_{self.out.id}')
        self.params["padding"] = model.new_int_var(0, MAX_KERNEL_SIZE, f'padding_{self.out.id}')
        self.params["dilation"] = model.new_int_var(1, MAX_KERNEL_SIZE, f'dilation_{self.out.id}')
        self.model.add(self.params["stride"] <= self.params["kernel_size"])
        self.model.add(2 * self.params["padding"] <= self.params["kernel_size"])

        for i in range(MAX_DIM):
            lt = self.model.new_bool_var("lt")
            self.model.add(i < MAX_DIM - self.params["spatial_dims"]).only_enforce_if(lt)
            self.model.add(i >= MAX_DIM - self.params["spatial_dims"]).only_enforce_if(~lt)
            self.model.add(self.out.shape[i] == self.in1.shape[i]).only_enforce_if(lt)
            '''
                d_out = floor((d_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1
            '''
            # Compute dilation * (kernel_size - 1)
            kernel_minus_one = model.new_int_var(0, MAX_KERNEL_SIZE - 1, f'kernel_minus_one_{self.out.id}_{i}')
            self.model.add(kernel_minus_one == self.params["kernel_size"] - 1)
            dilation_times_kernel_minus_one = model.new_int_var(0, MAX_KERNEL_SIZE * (MAX_KERNEL_SIZE - 1), f'dilation_times_kernel_minus_one_{self.out.id}_{i}')
            self.model.add_multiplication_equality(dilation_times_kernel_minus_one, [self.params["dilation"], kernel_minus_one])
            
            # Compute d_in + 2 * padding - dilation * (kernel_size - 1) - 1
            numerator = model.new_int_var(0, 2 * MAX_SIZE, f'numerator_{self.out.id}_{i}')
            self.model.add(numerator == self.in1.shape[i] + 2 * self.params["padding"] - dilation_times_kernel_minus_one - 1).only_enforce_if(~lt)
            div = model.new_int_var(0, MAX_SIZE, f'div_{self.out.id}_{i}')
            self.model.add_division_equality(div, numerator, self.params["stride"])
            self.model.add(self.out.shape[i] == div + 1).only_enforce_if(~lt)

        # Compute kernel_volume = kernel_size^spatial_dims
        # Since spatial_dims can be 1, 2, or 3, we use conditional logic
        kernel_squared = model.new_int_var(1, MAX_KERNEL_SIZE ** 2, f'kernel_squared_{self.out.id}')
        self.model.add_multiplication_equality(kernel_squared, [self.params["kernel_size"], self.params["kernel_size"]])
        
        kernel_cubed = model.new_int_var(1, MAX_KERNEL_SIZE ** 3, f'kernel_cubed_{self.out.id}')
        self.model.add_multiplication_equality(kernel_cubed, [kernel_squared, self.params["kernel_size"]])
        
        self.kernel_volume = model.new_int_var(1, MAX_KERNEL_SIZE ** 3, f'kernel_volume_{self.out.id}')
        
        is_dim1 = model.new_bool_var(f'is_dim1_{self.out.id}')
        is_dim2 = model.new_bool_var(f'is_dim2_{self.out.id}')
        is_dim3 = model.new_bool_var(f'is_dim3_{self.out.id}')
        
        self.model.add(self.params["spatial_dims"] == 1).only_enforce_if(is_dim1)
        self.model.add(self.params["spatial_dims"] != 1).only_enforce_if(~is_dim1)
        self.model.add(self.params["spatial_dims"] == 2).only_enforce_if(is_dim2)
        self.model.add(self.params["spatial_dims"] != 2).only_enforce_if(~is_dim2)
        self.model.add(self.params["spatial_dims"] == 3).only_enforce_if(is_dim3)
        self.model.add(self.params["spatial_dims"] != 3).only_enforce_if(~is_dim3)
        
        self.model.add(self.kernel_volume == self.params["kernel_size"]).only_enforce_if(is_dim1)
        self.model.add(self.kernel_volume == kernel_squared).only_enforce_if(is_dim2)
        self.model.add(self.kernel_volume == kernel_cubed).only_enforce_if(is_dim3)
        
        # Compute kernel_volume * output_size for FLOPS
        self.kernel_volume_times_output = model.new_int_var(1, MAX_SIZE_PRODUCT, f'kernel_vol_times_out_{self.out.id}')
        self.model.add_multiplication_equality(self.kernel_volume_times_output, [self.out.size, self.kernel_volume])

    def render(self, tensor_names: Dict[int, str]) -> str:
        kernel_arg = str(self.resolved_params["kernel_size"])
        stride_arg = str(self.resolved_params["stride"])
        padding_arg = str(self.resolved_params["padding"])
        dilation_arg = str(self.resolved_params["dilation"])
        return (
            f"    {tensor_names[self.out.id]} = "
            f"torch.nn.functional.max_pool{self.resolved_params['spatial_dims']}d("
            f"{tensor_names[self.in1.id]}, kernel_size={kernel_arg}, stride={stride_arg}, padding={padding_arg}, dilation={dilation_arg})"
        )

    def flop(self) -> cp_model.LinearExpr:
        return self.kernel_volume_times_output

class MaxPool1d(MaxPoolNd):
    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins, 1)

class MaxPool2d(MaxPoolNd):
    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins, 2)

class MaxPool3d(MaxPoolNd):
    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins, 3)

class ConvNd(BasicOp):
    def __init__(self, model, ins: List[Edge], nd: Optional[int] = None):
        super().__init__(model, ins)
        assert len(ins) == 2, "Conv requires one input and one weight"
        self.in1 = ins[0]
        self.weight = ins[1]
        self.params["spatial_dims"] = model.new_int_var(1, 3, f'spatial_dims_{self.out.id}')
        if nd is not None:
            self.model.add(self.params["spatial_dims"] == nd)
        self.params["out_channels"] = model.new_int_var(1, MAX_SIZE, f'out_ch_{self.out.id}')
        self.params["stride"] = model.new_int_var(1, MAX_KERNEL_SIZE, f'stride_{self.out.id}')
        self.params["padding"] = model.new_int_var(0, MAX_KERNEL_SIZE, f'padding_{self.out.id}')
        self.params["dilation"] = model.new_int_var(1, MAX_KERNEL_SIZE, f'dilation_{self.out.id}')
        self.params["groups"] = model.new_int_var(1, 32, f'groups_{self.out.id}')
        self.model.add(self.in1.num == self.params["spatial_dims"] + 2)
        self.model.add(self.out.num == self.params["spatial_dims"] + 2)
        self.model.add(self.weight.num == self.params["spatial_dims"] + 2)
        # Constraints for stride and padding
        self.model.add(2 * self.params["padding"] <= MAX_KERNEL_SIZE)
        
        # Constraints for groups: both in_channels and out_channels must be divisible by groups
        # The channel dimension is at index MAX_DIM - self.in1.num + 1
        in_channels_per_group = model.new_int_var(1, MAX_SIZE, f'in_channels_per_group_{self.out.id}')
        out_channels_per_group = model.new_int_var(1, MAX_SIZE, f'out_channels_per_group_{self.out.id}')
        in_channels_times_groups = model.new_int_var(1, MAX_SIZE_PRODUCT, f'in_channels_times_groups_{self.out.id}')
        out_channels_times_groups = model.new_int_var(1, MAX_SIZE_PRODUCT, f'out_channels_times_groups_{self.out.id}')
        self.model.add_multiplication_equality(in_channels_times_groups, [in_channels_per_group, self.params["groups"]])
        self.model.add_multiplication_equality(out_channels_times_groups, [out_channels_per_group, self.params["groups"]])
        # Ensure out_channels is divisible by groups
        self.model.add(self.params["out_channels"] == out_channels_times_groups)
        
        for i in range(MAX_DIM):
            is_channel_dim = model.new_bool_var(f"is_channel_dim_{self.out.id}_{i}")
            self.model.add(MAX_DIM - self.in1.num + 1 == i).only_enforce_if(is_channel_dim)
            self.model.add(MAX_DIM - self.in1.num + 1 != i).only_enforce_if(~is_channel_dim)
            # For the channel dimension, ensure in_channels is divisible by groups
            self.model.add(self.in1.shape[i] == in_channels_times_groups).only_enforce_if(is_channel_dim)
        
        prod = self.out.size

        for i in range(MAX_DIM):
            is_batch_index = self.model.new_bool_var("is_batch_index")
            is_channel_index = self.model.new_bool_var("is_channel_index")
            self.model.add(i == MAX_DIM - self.params["spatial_dims"] - 2).only_enforce_if(is_batch_index)
            self.model.add(i != MAX_DIM - self.params["spatial_dims"] - 2).only_enforce_if(~is_batch_index)
            self.model.add(i == MAX_DIM - self.params["spatial_dims"] - 1).only_enforce_if(is_channel_index)
            self.model.add(i != MAX_DIM - self.params["spatial_dims"] - 1).only_enforce_if(~is_channel_index)
            self.model.add(self.out.shape[i] == self.in1.shape[i]).only_enforce_if(is_batch_index)
            self.model.add(self.out.shape[i] == self.params["out_channels"]).only_enforce_if(is_channel_index)
            in_spatial_index = self.model.new_bool_var("in_spatial_index")
            kernel = model.new_int_var(MIN_KERNEL_SIZE, MAX_KERNEL_SIZE, f'kernel_{self.out.id}_{i}')
            self.model.add(i >= MAX_DIM - self.params["spatial_dims"]).only_enforce_if(in_spatial_index)
            self.model.add(i < MAX_DIM - self.params["spatial_dims"]).only_enforce_if(~in_spatial_index)
            self.model.add(self.in1.shape[i] >= kernel).only_enforce_if(in_spatial_index)
            
            # Update output shape calculation with stride, padding, and dilation
            # d_out = floor((d_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1
            # Compute dilation * (kernel_size - 1)
            kernel_minus_one = model.new_int_var(0, MAX_KERNEL_SIZE - 1, f'kernel_minus_one_{self.out.id}_{i}')
            self.model.add(kernel_minus_one == kernel - 1)
            dilation_times_kernel_minus_one = model.new_int_var(0, MAX_KERNEL_SIZE * (MAX_KERNEL_SIZE - 1), f'dilation_times_kernel_minus_one_{self.out.id}_{i}')
            self.model.add_multiplication_equality(dilation_times_kernel_minus_one, [self.params["dilation"], kernel_minus_one])
            
            # Compute d_in + 2 * padding - dilation * (kernel_size - 1) - 1
            numerator = model.new_int_var(0, 2 * MAX_SIZE, f'numerator_{self.out.id}_{i}')
            self.model.add(numerator == self.in1.shape[i] + 2 * self.params["padding"] - dilation_times_kernel_minus_one - 1)
            div = model.new_int_var(0, MAX_SIZE, f'div_{self.out.id}_{i}')
            self.model.add_division_equality(div, numerator, self.params["stride"])
            self.model.add(self.out.shape[i] == div + 1).only_enforce_if(in_spatial_index)
            
            self.params[f'kernel_{i}'] = kernel
            self.model.add(self.weight.shape[i] == kernel).only_enforce_if(in_spatial_index)
            self.model.add(self.weight.shape[i] == in_channels_per_group).only_enforce_if(is_channel_index)
            self.model.add(self.weight.shape[i] == self.params["out_channels"]).only_enforce_if(is_batch_index)

            next_prod = model.new_int_var(1, MAX_SIZE_PRODUCT, f'next_prod_{self.out.id}_{i}')
            new_prod = model.new_int_var(1, MAX_SIZE_PRODUCT, f'new_prod_{self.out.id}_{i}')
            new_prod_channel = model.new_int_var(1, MAX_SIZE_PRODUCT, f'new_prod_channel_{self.out.id}_{i}')
            self.model.add_multiplication_equality(new_prod, [prod, kernel])
            self.model.add_multiplication_equality(new_prod_channel, [prod, in_channels_per_group])
            '''
            if is_channel_index:
                next_prod = new_prod_channel
            elif in_spatial_index:
                next_prod = new_prod
            else:
                next_prod = prod
            
            is_channel_index => next_prod = new_prod_channel
            in_spatial_index => next_prod = new_prod
            ~in_spatial_index and ~is_channel_index => next_prod = prod
            '''

            self.model.add(next_prod == new_prod).only_enforce_if(in_spatial_index)
            self.model.add(next_prod == new_prod_channel).only_enforce_if(is_channel_index)
            q = self.model.new_bool_var("q")
            self.model.add_bool_or([in_spatial_index, is_channel_index, q])
            self.model.add(next_prod == prod).only_enforce_if(q)
            prod = next_prod

        self.conv_base = prod

    def render(self, tensor_names: Dict[int, str]) -> str:
        groups = self.resolved_params["groups"]
        # For grouped convolution, weight shape is [out_channels, in_channels // groups] + kernel_sizes
        
        stride_arg = str(self.resolved_params["stride"])
        padding_arg = str(self.resolved_params["padding"])
        dilation_arg = str(self.resolved_params["dilation"])
        groups_arg = str(groups)
        return (
            f"    {tensor_names[self.out.id]} = torch.nn.functional.conv{self.resolved_params['spatial_dims']}d("
            f"{tensor_names[self.in1.id]}, {tensor_names[self.weight.id]}, "
            f"stride={stride_arg}, padding={padding_arg}, dilation={dilation_arg}, groups={groups_arg})"
        )

    def flop(self) -> cp_model.LinearExpr:
        return 2 * self.conv_base

class Conv1d(ConvNd):
    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins, 1)

class Conv2d(ConvNd):
    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins, 2)

class Conv3d(ConvNd):
    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins, 3)

class ConvTransposeNd(BasicOp):
    def __init__(self, model, ins: List[Edge], nd: Optional[int] = None):
        super().__init__(model, ins)
        assert len(ins) == 2, "ConvTranspose requires one input and one weight"
        self.in1 = ins[0]
        self.weight = ins[1]
        self.params["spatial_dims"] = model.new_int_var(1, 3, f'spatial_dims_{self.out.id}')
        if nd is not None:
            self.model.add(self.params["spatial_dims"] == nd)
        self.params["out_channels"] = model.new_int_var(1, MAX_SIZE, f'out_ch_{self.out.id}')
        self.params["stride"] = model.new_int_var(1, MAX_KERNEL_SIZE, f'stride_{self.out.id}')
        self.params["padding"] = model.new_int_var(0, MAX_KERNEL_SIZE, f'padding_{self.out.id}')
        self.params["output_padding"] = model.new_int_var(0, MAX_KERNEL_SIZE - 1, f'output_padding_{self.out.id}')
        self.params["dilation"] = model.new_int_var(1, MAX_KERNEL_SIZE, f'dilation_{self.out.id}')
        self.params["groups"] = model.new_int_var(1, 32, f'groups_{self.out.id}')
        self.model.add(self.in1.num == self.params["spatial_dims"] + 2)
        self.model.add(self.out.num == self.params["spatial_dims"] + 2)
        self.model.add(self.weight.num == self.params["spatial_dims"] + 2)
        # Constraints for stride and padding
        self.model.add(2 * self.params["padding"] <= MAX_KERNEL_SIZE)
        # Constraint: output_padding must be < stride
        self.model.add(self.params["output_padding"] < self.params["stride"])
        
        # Constraints for groups: both in_channels and out_channels must be divisible by groups
        # The channel dimension is at index MAX_DIM - self.in1.num + 1
        in_channels_per_group = model.new_int_var(1, MAX_SIZE, f'in_channels_per_group_{self.out.id}')
        out_channels_per_group = model.new_int_var(1, MAX_SIZE, f'out_channels_per_group_{self.out.id}')
        in_channels_times_groups = model.new_int_var(1, MAX_SIZE_PRODUCT, f'in_channels_times_groups_{self.out.id}')
        out_channels_times_groups = model.new_int_var(1, MAX_SIZE_PRODUCT, f'out_channels_times_groups_{self.out.id}')
        self.model.add_multiplication_equality(in_channels_times_groups, [in_channels_per_group, self.params["groups"]])
        self.model.add_multiplication_equality(out_channels_times_groups, [out_channels_per_group, self.params["groups"]])
        # Ensure out_channels is divisible by groups
        self.model.add(self.params["out_channels"] == out_channels_times_groups)
        
        for i in range(MAX_DIM):
            is_channel_dim = model.new_bool_var(f"is_channel_dim_{self.out.id}_{i}")
            self.model.add(MAX_DIM - self.in1.num + 1 == i).only_enforce_if(is_channel_dim)
            self.model.add(MAX_DIM - self.in1.num + 1 != i).only_enforce_if(~is_channel_dim)
            # For the channel dimension, ensure in_channels is divisible by groups
            self.model.add(self.in1.shape[i] == in_channels_times_groups).only_enforce_if(is_channel_dim)
        
        prod = self.out.size

        for i in range(MAX_DIM):
            is_batch_index = self.model.new_bool_var("is_batch_index")
            is_channel_index = self.model.new_bool_var("is_channel_index")
            self.model.add(i == MAX_DIM - self.params["spatial_dims"] - 2).only_enforce_if(is_batch_index)
            self.model.add(i != MAX_DIM - self.params["spatial_dims"] - 2).only_enforce_if(~is_batch_index)
            self.model.add(i == MAX_DIM - self.params["spatial_dims"] - 1).only_enforce_if(is_channel_index)
            self.model.add(i != MAX_DIM - self.params["spatial_dims"] - 1).only_enforce_if(~is_channel_index)
            self.model.add(self.out.shape[i] == self.in1.shape[i]).only_enforce_if(is_batch_index)
            self.model.add(self.out.shape[i] == self.params["out_channels"]).only_enforce_if(is_channel_index)
            in_spatial_index = self.model.new_bool_var("in_spatial_index")
            kernel = model.new_int_var(MIN_KERNEL_SIZE, MAX_KERNEL_SIZE, f'kernel_{self.out.id}_{i}')
            self.model.add(i >= MAX_DIM - self.params["spatial_dims"]).only_enforce_if(in_spatial_index)
            self.model.add(i < MAX_DIM - self.params["spatial_dims"]).only_enforce_if(~in_spatial_index)
            
            # Update output shape calculation for transposed convolution
            # d_out = (d_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
            # Compute (d_in - 1) * stride
            input_minus_one = model.new_int_var(0, MAX_SIZE - 1, f'input_minus_one_{self.out.id}_{i}')
            self.model.add(input_minus_one == self.in1.shape[i] - 1)
            input_minus_one_times_stride = model.new_int_var(0, MAX_SIZE * MAX_KERNEL_SIZE, f'input_minus_one_times_stride_{self.out.id}_{i}')
            self.model.add_multiplication_equality(input_minus_one_times_stride, [input_minus_one, self.params["stride"]])
            
            # Compute dilation * (kernel_size - 1)
            kernel_minus_one = model.new_int_var(0, MAX_KERNEL_SIZE - 1, f'kernel_minus_one_{self.out.id}_{i}')
            self.model.add(kernel_minus_one == kernel - 1)
            dilation_times_kernel_minus_one = model.new_int_var(0, MAX_KERNEL_SIZE * (MAX_KERNEL_SIZE - 1), f'dilation_times_kernel_minus_one_{self.out.id}_{i}')
            self.model.add_multiplication_equality(dilation_times_kernel_minus_one, [self.params["dilation"], kernel_minus_one])
            
            # Compute (d_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
            output_size = model.new_int_var(1, MAX_SIZE, f'output_size_{self.out.id}_{i}')
            self.model.add(output_size == input_minus_one_times_stride - 2 * self.params["padding"] + dilation_times_kernel_minus_one + self.params["output_padding"] + 1)
            self.model.add(self.out.shape[i] == output_size).only_enforce_if(in_spatial_index)
            
            self.params[f'kernel_{i}'] = kernel
            self.model.add(self.weight.shape[i] == kernel).only_enforce_if(in_spatial_index)
            self.model.add(self.weight.shape[i] == out_channels_per_group).only_enforce_if(is_channel_index)
            self.model.add(self.weight.shape[i] == in_channels_times_groups).only_enforce_if(is_batch_index)

            next_prod = model.new_int_var(1, MAX_SIZE_PRODUCT, f'next_prod_{self.out.id}_{i}')
            new_prod = model.new_int_var(1, MAX_SIZE_PRODUCT, f'new_prod_{self.out.id}_{i}')
            new_prod_channel = model.new_int_var(1, MAX_SIZE_PRODUCT, f'new_prod_channel_{self.out.id}_{i}')
            self.model.add_multiplication_equality(new_prod, [prod, kernel])
            self.model.add_multiplication_equality(new_prod_channel, [prod, in_channels_per_group])
            
            self.model.add(next_prod == new_prod).only_enforce_if(in_spatial_index)
            self.model.add(next_prod == new_prod_channel).only_enforce_if(is_channel_index)
            q = self.model.new_bool_var("q")
            self.model.add_bool_or([in_spatial_index, is_channel_index, q])
            self.model.add(next_prod == prod).only_enforce_if(q)
            prod = next_prod

        self.conv_base = prod

    def render(self, tensor_names: Dict[int, str]) -> str:
        groups = self.resolved_params["groups"]
        # For transposed grouped convolution, weight shape is [in_channels, out_channels // groups] + kernel_sizes

        stride_arg = str(self.resolved_params["stride"])
        padding_arg = str(self.resolved_params["padding"])
        output_padding_arg = str(self.resolved_params["output_padding"])
        dilation_arg = str(self.resolved_params["dilation"])
        groups_arg = str(groups)
        return (
            f"    {tensor_names[self.out.id]} = torch.nn.functional.conv_transpose{self.resolved_params['spatial_dims']}d("
            f"{tensor_names[self.in1.id]}, {tensor_names[self.weight.id]}, "
            f"stride={stride_arg}, padding={padding_arg}, output_padding={output_padding_arg}, "
            f"groups={groups_arg}, dilation={dilation_arg})"
        )

    def flop(self) -> cp_model.LinearExpr:
        return 2 * self.conv_base

class ConvTranspose1d(ConvTransposeNd):
    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins, 1)

class ConvTranspose2d(ConvTransposeNd):
    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins, 2)

class ConvTranspose3d(ConvTransposeNd):
    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins, 3)

# Unary operations
class Cos(UnaryOp):
    torch_func = "torch.cos"
    flop_multiplier = 1

class Sin(UnaryOp):
    torch_func = "torch.sin"
    flop_multiplier = 1

class Exp2(UnaryOp):
    torch_func = "torch.exp2"
    flop_multiplier = 1

class LogSigmoid(UnaryOp):
    torch_func = "torch.nn.functional.logsigmoid"
    flop_multiplier = 4

class Abs(UnaryOp):
    torch_func = "torch.abs"
    flop_multiplier = 1

class Maximum(ElementwiseOp):
    torch_func = "torch.maximum"

class Minimum(ElementwiseOp):
    torch_func = "torch.minimum"

class Lerp(BasicOp):
    torch_func = "torch.lerp"
    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins)
        assert len(ins) == 3, "Lerp requires 3 inputs (start, end, weight)"
        self.in1, self.in2, self.in3 = ins
        # Lerp output shape matches start/end (broadcasted)
        for i in range(MAX_DIM):
            eq1 = self.model.new_bool_var("eq0")
            eq2 = self.model.new_bool_var("eq1")
            eq3 = self.model.new_bool_var("eq2")
            in1_eq_1 = self.model.new_bool_var("in1_eq_1")
            in2_eq_1 = self.model.new_bool_var("in2_eq_1")
            in3_eq_1 = self.model.new_bool_var("in3_eq_1")
            
            self.model.add(self.in1.shape[i] == self.in2.shape[i]).only_enforce_if(eq1)
            self.model.add(self.in1.shape[i] != self.in2.shape[i]).only_enforce_if(~eq1)
            self.model.add(self.in1.shape[i] == self.in3.shape[i]).only_enforce_if(eq2)
            self.model.add(self.in1.shape[i] != self.in3.shape[i]).only_enforce_if(~eq2)
            self.model.add(self.in2.shape[i] == self.in3.shape[i]).only_enforce_if(eq3)
            self.model.add(self.in2.shape[i] != self.in3.shape[i]).only_enforce_if(~eq3)
            self.model.add(self.in1.shape[i] == 1).only_enforce_if(in1_eq_1)
            self.model.add(self.in1.shape[i] != 1).only_enforce_if(~in1_eq_1)
            self.model.add(self.in2.shape[i] == 1).only_enforce_if(in2_eq_1)
            self.model.add(self.in2.shape[i] != 1).only_enforce_if(~in2_eq_1)
            self.model.add(self.in3.shape[i] == 1).only_enforce_if(in3_eq_1)
            self.model.add(self.in3.shape[i] != 1).only_enforce_if(~in3_eq_1)
            # or1 = in1_eq_1 or in2_eq_1 or eq1,
            # or2 = in1_eq_1 or in3_eq_1 or eq2,
            # or3 = in2_eq_1 or in3_eq_1 or eq3,
            or1 = self.model.new_bool_var("or1")
            or2 = self.model.new_bool_var("or2")
            or3 = self.model.new_bool_var("or3")
            self.model.add_bool_or([~or1, in1_eq_1, in2_eq_1, eq1])
            self.model.add(or1 == True).only_enforce_if(eq1)
            self.model.add(or1 == True).only_enforce_if(in1_eq_1)
            self.model.add(or1 == True).only_enforce_if(in2_eq_1)
            self.model.add_bool_or([~or2, in1_eq_1, in3_eq_1, eq2])
            self.model.add(or2 == True).only_enforce_if(eq2)
            self.model.add(or2 == True).only_enforce_if(in1_eq_1)
            self.model.add(or2 == True).only_enforce_if(in3_eq_1)
            self.model.add_bool_or([~or3, in2_eq_1, in3_eq_1, eq3])
            self.model.add(or3 == True).only_enforce_if(eq3)
            self.model.add(or3 == True).only_enforce_if(in2_eq_1)
            self.model.add(or3 == True).only_enforce_if(in3_eq_1)
            self.model.add_bool_and([or1, or2, or3])
            self.model.add_max_equality(self.out.shape[i], [self.in1.shape[i], self.in2.shape[i], self.in3.shape[i]])

        self.model.add_max_equality(self.out.num, [self.in1.num, self.in2.num, self.in3.num])
    
    def render(self, tensor_names: Dict[int, str]) -> str:
        return f"    {tensor_names[self.out.id]} = torch.lerp({tensor_names[self.in1.id]}, {tensor_names[self.in2.id]}, {tensor_names[self.in3.id]})"
    
    def flop(self) -> cp_model.LinearExpr:
        # Lerp: start + weight * (end - start) = 1 sub + 1 mul + 1 add = 3 FLOPs per element
        return 3 * self.out.size


class Var(ReduceOp):
    torch_func = "torch.var"
    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins)
        
    def flop(self) -> cp_model.LinearExpr:
        # Variance: mean (N ops) + (x-mean)^2 for each element (2N ops: sub + mul) = 3N ops
        return self.in1.size * 3  

class Norm(ReduceOp):
    torch_func = "torch.norm"
    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins)
        
    def flop(self) -> cp_model.LinearExpr:
        # Norm: square each element (1 op) + sum (1 op) + sqrt (1 op) = 3 ops per element
        return self.in1.size * 3

# Cumulative operations
class CumMax(UnaryOpWithDim):
    torch_func = "torch.cummax"
    flop_multiplier = 1

    def render(self, tensor_names: Dict[int, str]) -> str:
        return (
            f"    {tensor_names[self.out.id]} = torch.cummax("
            f"{tensor_names[self.in1.id]}, dim = {self.resolved_params['dim']}).values"
        )

class CumMin(UnaryOpWithDim):
    torch_func = "torch.cummin"
    flop_multiplier = 1

    def render(self, tensor_names: Dict[int, str]) -> str:
        return (
            f"    {tensor_names[self.out.id]} = torch.cummin("
            f"{tensor_names[self.in1.id]}, dim = {self.resolved_params['dim']}).values"
        )

class CumSum(UnaryOpWithDim):
    torch_func = "torch.cumsum"
    flop_multiplier = 1

# Matrix operations
class Bmm(BasicOp):
    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins)
        assert len(ins) == 2, "Bmm requires 2 inputs"
        self.in1, self.in2 = ins
        self.model.add(self.in1.num == 3)
        self.model.add(self.in2.num == 3)
        self.model.add(self.out.num == 3)
        self.model.add(self.in1.shape[-3] == self.in2.shape[-3])
        self.model.add(self.in1.shape[-1] == self.in2.shape[-2])
        self.model.add(self.out.shape[-3] == self.in1.shape[-3])
        self.model.add(self.out.shape[-2] == self.in1.shape[-2])
        self.model.add(self.out.shape[-1] == self.in2.shape[-1])
        self.product = self.model.new_int_var(1, MAX_SIZE_PRODUCT, f'product_{self.out.id}')
        self.model.add_multiplication_equality(self.product, [self.out.size, self.in1.shape[-1]])

    def render(self, tensor_names: Dict[int, str]) -> str:
        return f"    {tensor_names[self.out.id]} = torch.bmm({tensor_names[self.in1.id]}, {tensor_names[self.in2.id]})"

    def flop(self) -> cp_model.LinearExpr:
        return 2 * self.product

class Triu(UnaryOp):
    torch_func = "torch.triu"
    flop_multiplier = 0  # Just masking/selection, no computation
    
    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins)
        # triu operates on the last two dimensions, so input must be at least 2D
        self.model.add(self.in1.num >= 2)

class Tril(UnaryOp):
    torch_func = "torch.tril"
    flop_multiplier = 0  # Just masking/selection, no computation
    
    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins)
        # tril operates on the last two dimensions, so input must be at least 2D
        self.model.add(self.in1.num >= 2)

# Concatenation/Stacking operations
class Cat(BasicOp):
    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins)
        assert len(ins) >= 2, "Cat requires at least 2 inputs"
        self.params["dim"] = model.new_int_var(0, MAX_DIM - 1, f'dim_{self.out.id}')
        # All inputs must have same number of dimensions
        for i in range(1, len(ins)):
            self.model.add(ins[0].num == ins[i].num)
        # All dimensions except the concatenation dimension must match
        for d in range(MAX_DIM):
            is_cat_dim = self.model.new_bool_var(f"is_cat_dim_{self.out.id}_{d}")
            self.model.add(self.params["dim"] + MAX_DIM - ins[0].num == d).only_enforce_if(is_cat_dim)
            self.model.add(self.params["dim"] + MAX_DIM - ins[0].num != d).only_enforce_if(~is_cat_dim)
            # For non-cat dimensions, all inputs must have same size
            for i in range(1, len(ins)):
                self.model.add(ins[0].shape[d] == ins[i].shape[d]).only_enforce_if(~is_cat_dim)
            # For cat dimension, output is sum of all input sizes
            
            total_size = 0
            for i in range(len(ins)):
                total_size += ins[i].shape[d]
            self.model.add(self.out.shape[d] == total_size).only_enforce_if(is_cat_dim)
            self.model.add(self.out.shape[d] == ins[0].shape[d]).only_enforce_if(~is_cat_dim)
        self.model.add(self.out.num == ins[0].num)
        self.model.add(self.params["dim"] < ins[0].num)

    def render(self, tensor_names: Dict[int, str]) -> str:
        args = ", ".join(tensor_names[edge.id] for edge in self.ins)
        dim = self.resolved_params["dim"]
        return f"    {tensor_names[self.out.id]} = torch.cat([{args}], dim={dim})"

    def flop(self) -> cp_model.LinearExpr:
        return 0

class Stack(BasicOp):
    def __init__(self, model, ins: List[Edge]):
        super().__init__(model, ins)
        assert len(ins) >= 2, "Stack requires at least 2 inputs"
        self.params["dim"] = model.new_int_var(0, MAX_DIM - 1, f'dim_{self.out.id}')
        for i in range(1, len(ins)):
            self.model.add(ins[0].num == ins[i].num)
            for d in range(MAX_DIM):
                self.model.add(ins[0].shape[d] == ins[i].shape[d])
        
        '''
            out[i] = len(ins), if dim == i
            out[i] = in[i+1], if dim > i
            out[i] = in[i], if dim < i
        '''
        # Other dimensions match input
        for i in range(MAX_DIM):
            eqi = self.model.new_bool_var("eqi")
            gti = self.model.new_bool_var("gti")
            lti = self.model.new_bool_var("lti")
            self.model.add(self.params["dim"] + MAX_DIM - ins[0].num - 1 == i).only_enforce_if(eqi)
            self.model.add(self.params["dim"] + MAX_DIM - ins[0].num - 1 != i).only_enforce_if(~eqi)
            self.model.add(self.params["dim"] + MAX_DIM - ins[0].num - 1 > i).only_enforce_if(gti)
            self.model.add(self.params["dim"] + MAX_DIM - ins[0].num - 1 <= i).only_enforce_if(~gti)
            self.model.add(self.params["dim"] + MAX_DIM - ins[0].num - 1 < i).only_enforce_if(lti)
            self.model.add(self.params["dim"] + MAX_DIM - ins[0].num - 1 >= i).only_enforce_if(~lti)
            self.model.add(self.out.shape[i] == len(ins)).only_enforce_if(eqi)
            if i < MAX_DIM - 1:
                self.model.add(self.out.shape[i] == ins[0].shape[i + 1]).only_enforce_if(gti)
            self.model.add(self.out.shape[i] == ins[0].shape[i]).only_enforce_if(lti)
        
        self.model.add(self.params["dim"] <= ins[0].num)
        self.model.add(self.out.num == ins[0].num + 1)

    def render(self, tensor_names: Dict[int, str]) -> str:
        args = ", ".join(tensor_names[edge.id] for edge in self.ins)
        dim = self.resolved_params["dim"]
        return f"    {tensor_names[self.out.id]} = torch.stack([{args}], dim={dim})"

    def flop(self) -> cp_model.LinearExpr:
        return 0

# Constant generation operations (similar to Randn)
class Ones(BasicOp):
    def __init__(self, model):
        super().__init__(model, [], torch.float32)
    
    def render_input(self, tensor_names: Dict[int, str]) -> str:
        dtype_name = format_dtype(self.out.dtype)
        dtype_arg = f", dtype={dtype_name}" if dtype_name else ""
        return f"    {tensor_names[self.out.id]} = torch.ones({self.out.shape}{dtype_arg})"

    def flop(self) -> cp_model.LinearExpr:
        return 0

class Zeros(BasicOp):
    def __init__(self, model):
        super().__init__(model, [], torch.float32)
    
    def render_input(self, tensor_names: Dict[int, str]) -> str:
        dtype_name = format_dtype(self.out.dtype)
        dtype_arg = f", dtype={dtype_name}" if dtype_name else ""
        return f"    {tensor_names[self.out.id]} = torch.zeros({self.out.shape}{dtype_arg})"

    def flop(self) -> cp_model.LinearExpr:
        return 0