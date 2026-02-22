import random
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Type

from ortools.sat.python import cp_model

from ops import *

@dataclass(frozen=True)
class OpSpec:
    op_cls: Type[BasicOp]
    num_inputs: int


DEFAULT_OP_SPECS: List[OpSpec] = [
    OpSpec(Add, 2),
    OpSpec(Sub, 2),
    OpSpec(Mul, 2),
    OpSpec(Div, 2),
    OpSpec(Matmul, 2),
    OpSpec(Sum, 1),
    OpSpec(Mean, 1),
    OpSpec(Max, 1),
    OpSpec(Min, 1),
    OpSpec(ArgMax, 1),
    OpSpec(ArgMin, 1),
    OpSpec(Transpose, 1),
    OpSpec(ReLU, 1),
    OpSpec(LeakyReLU, 1),
    OpSpec(Sigmoid, 1),
    OpSpec(Tanh, 1),
    OpSpec(Softmax, 1),
    OpSpec(LogSoftmax, 1),
    OpSpec(Swish, 1),
    OpSpec(GELU, 1),
    OpSpec(SELU, 1),
    OpSpec(Hardsigmoid, 1),
    OpSpec(Softplus, 1),
    OpSpec(Softsign, 1),
    OpSpec(ELU, 1),
    OpSpec(HardTanh, 1),
    OpSpec(Clamp, 1),
    OpSpec(BatchNorm, 1),
    OpSpec(LayerNorm, 1),
    OpSpec(GroupNorm, 1),
    OpSpec(InstanceNorm, 1),
    OpSpec(AvgPool1d, 1),
    OpSpec(AvgPool2d, 1),
    OpSpec(AvgPool3d, 1),
    OpSpec(MaxPool1d, 1),
    OpSpec(MaxPool2d, 1),
    OpSpec(MaxPool3d, 1),
    OpSpec(Conv1d, 2),
    OpSpec(Conv2d, 2),
    OpSpec(Conv3d, 2),
    OpSpec(ConvTranspose1d, 2),
    OpSpec(ConvTranspose2d, 2),
    OpSpec(ConvTranspose3d, 2),
    OpSpec(Cos, 1),
    OpSpec(Sin, 1),
    OpSpec(Exp2, 1),
    OpSpec(LogSigmoid, 1),
    OpSpec(Abs, 1),
    OpSpec(Maximum, 2),
    OpSpec(Minimum, 2),
    OpSpec(Lerp, 3),
    OpSpec(Var, 1),
    OpSpec(Norm, 1),
    OpSpec(CumMax, 1),
    OpSpec(CumMin, 1),
    OpSpec(CumSum, 1),
    OpSpec(Bmm, 2),
    OpSpec(Triu, 1),
    OpSpec(Tril, 1),
    OpSpec(Cat, 2),
    OpSpec(Stack, 2)
]

def nodes_to_op_specs(nodes: List[Type[BasicOp]]) -> List[OpSpec]:
    """
    Convert a list of node classes to a list of OpSpec objects from DEFAULT_OP_SPECS.
    
    Args:
        nodes: List of node classes (e.g., [CumSum, Softmax, Stack, ReLU, Stack])
    
    Returns:
        List of OpSpec objects corresponding to the input node classes
    
    Raises:
        ValueError: If a node class is not found in DEFAULT_OP_SPECS
    """
    # Create a mapping from op class to OpSpec
    op_spec_map = {spec.op_cls: spec for spec in DEFAULT_OP_SPECS}
    
    result = []
    for node_cls in nodes:
        if node_cls not in op_spec_map:
            raise ValueError(f"Node class {node_cls.__name__} not found in DEFAULT_OP_SPECS")
        result.append(op_spec_map[node_cls])
    
    return result

class Graph:
    def __init__(
        self,
        op_specs: Optional[Sequence[OpSpec]] = None,
        min_flops: Optional[int] = None,
        max_flops: Optional[int] = None,
        random_seed: Optional[int] = None,
        print_graph: bool = False,
    ):
        self.model = cp_model.CpModel()
        self.nodes: List[BasicOp] = []
        self.edges: Dict[int, Edge] = {}
        self.pool: List[Edge] = []
        self.op_specs = list(op_specs) if op_specs else DEFAULT_OP_SPECS
        self.min_flops = min_flops
        self.max_flops = max_flops
        self.random_seed = random_seed if random_seed is not None else random.randint(0, 2**31 - 1)
        self._solver: Optional[cp_model.CpSolver] = None
        self._solved = False
        self.total_flops = self.model.new_constant(0)
        self.total_memory = self.model.new_constant(0)
        self.print_graph = print_graph
    
    def _clear(self) -> None:
        Edge._next_id = 0 #TODO: This is a hack to reset the edge id. We should find a better way to do this.
        self.nodes = []
        self.edges = {}
        self.pool = []
        self.total_flops = self.model.new_constant(0)
        self.total_memory = self.model.new_constant(0)
        self._solver = None
        self._solved = False

    def _register_edge(self, edge: Edge) -> None:
        self.edges[edge.id] = edge
        edge.users = []

    def _register_node(self, op: BasicOp) -> None:
        self.nodes.append(op)
        self._register_edge(op.out)
        self.pool.append(op.out)
        self.total_flops += op.flop()
        self.total_memory += op.out.size

    def _select_inputs(self, count: int) -> List[Edge]:
        if count <= 0 or not self.pool:
            return []
        pool = list(self.pool)
        # Uniform random selection without replacement
        selected = random.sample(pool, min(count, len(pool)))
        # Remove selected inputs from pool (each tensor used exactly once)
        for edge in selected:
            self.pool.remove(edge)
        return selected

    def generate_random_graph(self, num_nodes: int, node_op_types: Optional[List[Type[BasicOp]]] = None) -> None:
        self._clear()
        tensor_generators = [Randn]
        # tensor_generators = [Randn, Ones, Zeros]
        # We find that zeros will make the problem invaild, e.g., division by zero. And ones will make the problem too easy.
        # So we only use Randn for now.
        if node_op_types:
            assert len(node_op_types) == num_nodes, "Number of node op types must match number of nodes"
            node_op_types = nodes_to_op_specs(node_op_types) # type: ignore

        for _ in range(num_nodes):
            if node_op_types:
                spec = node_op_types[_]
            else:
                spec = random.choice(self.op_specs)
            while spec.num_inputs > len(self.pool):
                generator_cls = random.choice(tensor_generators)
                self._register_node(generator_cls(self.model))
            
            # Select inputs (this will remove them from pool)
            inputs = self._select_inputs(spec.num_inputs) if spec.num_inputs else []
            node = spec.op_cls(self.model, inputs)
            for edge in inputs:
                edge.users.append(node)
            # Register the node (adds output to pool)
            self._register_node(node)
        
        if self.min_flops is not None:
            self.model.add(self.total_flops >= self.min_flops)
        if self.max_flops is not None:
            self.model.add(self.total_flops <= self.max_flops)
        
        self.model.add(self.total_memory <= MAX_SIZE_TENSOR * 4)
        
        self.determine_shapes()

    def determine_shapes(self) -> None:
        solver = cp_model.CpSolver()
        solver.parameters.search_branching = cp_model.RANDOMIZED_SEARCH
        solver.parameters.random_seed = self.random_seed
        solver.parameters.enumerate_all_solutions = False
        solver.parameters.max_time_in_seconds = 10.0
        status = solver.solve(self.model)
        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            if self.print_graph:
                self.print_graph_structure()
            raise ValueError("Not feasible")

        self._solver = solver
        for edge in self.edges.values():
            edge.determine_shape(solver)
        self._materialize_params(solver)
        self._solved = True

    def _materialize_params(self, solver: cp_model.CpSolver) -> None:
        for node in self.nodes:
            resolved: Dict[str, object] = {}
            for key, value in node.params.items():
                resolved[key] = self._resolve_var(value, solver)
            node.resolved_params = resolved

    def _resolve_var(self, value: object, solver: cp_model.CpSolver) -> object:
        return solver.value(value)

    def flop_stats(self):
        if not self._solved:
            self.determine_shapes()
        if self._solver is None:
            raise ValueError("Solver not available")
        stats = []
        for node in self.nodes:
            flop_value = self._solver.Value(node.flop())
            stats.append((node, flop_value))
        return stats, self._solver.Value(self.total_flops)

    @staticmethod
    def _tensor_name(edge: Edge) -> str:
        return f"tensor_{edge.id}"

    def _final_outputs(self) -> List[Edge]:
        outputs = [
            edge for edge in self.edges.values()
            if not getattr(edge, "users", []) and not edge.tmp
        ]
        if not outputs and self.nodes:
            outputs = [self.nodes[-1].out]
        return outputs

    def print_graph_structure(self) -> None:
        """Print the graph structure for debugging purposes."""
        print("GRAPH STRUCTURE (Not Feasible Case):")
        for node in self.nodes:
            print(repr(node))

    def _input_nodes(self) -> List[BasicOp]:
        return [node for node in self.nodes if isinstance(node, (Randn, Ones, Zeros))]

    def _tensor_name_map(self) -> Dict[int, str]:
        return {node.out.id: self._tensor_name(node.out) for node in self.nodes}

    def io_tensors(self):
        if not self._solved:
            self.determine_shapes()
        inputs = self._input_nodes()
        outputs = self._final_outputs()
        if not outputs and self.nodes:
            outputs = [self.nodes[-1].out]
        return inputs, outputs, self._tensor_name_map()

    def to_torch(self) -> str:
        if not self._solved:
            self.determine_shapes()

        input_nodes, outputs, tensor_names = self.io_tensors()
        input_names = [tensor_names[node.out.id] for node in input_nodes]

        lines: List[str] = ["import torch", "", "def get_inputs():"]
        if input_nodes:
            for node in input_nodes:
                input_line = node.render_input(tensor_names)
                if input_line:
                    lines.append(input_line)
            lines.append(f"    return [{', '.join(input_names)}]")
        else:
            lines.append("    return []")
        lines.extend(["", f"def fused_operator({', '.join(input_names)}):"])

        for node in self.nodes:
            if isinstance(node, (Randn, Ones, Zeros)):
                continue
            line = node.render(tensor_names)
            if line:
                lines.append(line)

        output_names = [tensor_names[edge.id] for edge in outputs]
        lines.append(f"    return [{', '.join(output_names)}]")
        return "\n".join(lines)