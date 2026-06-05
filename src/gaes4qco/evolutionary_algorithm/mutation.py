import random
import math
from typing import List, Optional

from .interfaces import IMutationStrategy, IMutationPopulation
from .population import Population
from quantum_circuit.circuit import Circuit, Column
from quantum_circuit.gate_factory import GateFactory
from optimization.interfaces import IFitnessEvaluator


def _reset_evaluation_flags(circuit: Circuit):
    """Reseta os flags de um circuito para forçar a reavaliação e recalcular caches."""
    circuit.fitness = None
    circuit.fidelity = None


class RandomMutationSelector(IMutationPopulation):
    def __init__(self, mutation_strategies: List[IMutationStrategy], mutation_rate: float = 0.1):
        self._strategies = mutation_strategies
        self.mutation_rate = mutation_rate

    def mutate(self, population: Population) -> Population:
        mutated_individuals = []
        for circuit in population.get_individuals():
            if random.random() < self.mutation_rate:
                individual_copy = circuit.copy()
                applicable_strategies = [s for s in self._strategies if s.can_apply(individual_copy)]
                if applicable_strategies:
                    strategy: IMutationStrategy = random.choice(applicable_strategies)
                    mutated_circuit = strategy.mutate_individual(individual_copy)
                    _reset_evaluation_flags(mutated_circuit)
                    mutated_individuals.append(mutated_circuit)
                else:
                    mutated_individuals.append(circuit)
            else:
                mutated_individuals.append(circuit)
        return Population(mutated_individuals)


class BanditMutationSelector(IMutationPopulation):
    def __init__(
        self,
        mutation_strategies: List[IMutationStrategy],
        mutation_rate: float,
        fitness_evaluator: IFitnessEvaluator
    ):
        self._strategies = mutation_strategies
        self.mutation_rate = mutation_rate
        self._fitness_evaluator = fitness_evaluator
        self._rewards = {s.__class__.__name__: 0.0 for s in self._strategies}
        self._counts = {s.__class__.__name__: 0 for s in self._strategies}
        self._total_applications = 0

    def _select_strategy(self, individual: Circuit) -> IMutationStrategy:
        applicable_strategies = [s for s in self._strategies if s.can_apply(individual)]
        if not applicable_strategies:
            raise RuntimeError("Nenhuma estratégia de mutação aplicável encontrada.")

        untried_strategies = [s for s in applicable_strategies if self._counts[s.__class__.__name__] == 0]
        if untried_strategies:
            return random.choice(untried_strategies)

        n = self._total_applications
        ucb_scores = {}
        for s in applicable_strategies:
            name = s.__class__.__name__
            avg_reward = self._rewards[name] / self._counts[name]
            exploration_term = math.sqrt((2 * math.log(n)) / self._counts[name])
            ucb_scores[name] = avg_reward + exploration_term

        best_strategy_name = max(ucb_scores, key=ucb_scores.get)
        return next(s for s in applicable_strategies if s.__class__.__name__ == best_strategy_name)

    def mutate(self, population: Population) -> Population:
        mutated_individuals = []
        for circuit in population.get_individuals():
            if random.random() < self.mutation_rate:
                individual_copy = circuit.copy()
                strategy = self._select_strategy(individual_copy)

                original_fitness, _ = self._fitness_evaluator.evaluate(individual_copy)
                
                mutated_circuit = strategy.mutate_individual(individual_copy)

                # 4. Avalia o filho agora para alimentar o Bandit
                mutated_fitness, mutated_fidelity = self._fitness_evaluator.evaluate(mutated_circuit)

                # 5. SALVA o resultado dentro do circuito para que o Optimizer não calcule de novo!
                mutated_circuit.fitness = mutated_fitness
                mutated_circuit.fidelity = mutated_fidelity

                # Atualiza a IA
                reward = mutated_fitness - original_fitness
                strategy_name = strategy.__class__.__name__
                self._counts[strategy_name] += 1
                self._rewards[strategy_name] += reward
                self._total_applications += 1

                mutated_individuals.append(mutated_circuit)
            else:
                mutated_individuals.append(circuit)
        return Population(mutated_individuals)


class SwapColumnsMutation(IMutationStrategy):
    def can_apply(self, circuit: Circuit) -> bool:
        return circuit.depth > 1

    def mutate_individual(self, circuit: Circuit) -> Circuit:
        col1_idx, col2_idx = random.sample(range(circuit.depth), 2)
        circuit.columns[col1_idx], circuit.columns[col2_idx] = \
            circuit.columns[col2_idx], circuit.columns[col1_idx]
        return circuit


class SingleGateFlipMutation(IMutationStrategy):
    def __init__(self, gate_factory: GateFactory, use_evolutionary_strategy: bool):
        self._gate_factory = gate_factory
        self.use_evolutionary_strategy = use_evolutionary_strategy

    def can_apply(self, circuit: Circuit) -> bool:
        return any(col.gates for col in circuit.columns)

    def mutate_individual(self, circuit: Circuit) -> Circuit:
        non_empty_cols = [(i, col) for i, col in enumerate(circuit.columns) if col.gates]
        col_idx, target_col = random.choice(non_empty_cols)
        gate_idx_to_remove = random.randrange(len(target_col.gates))
        removed_gate = target_col.gates.pop(gate_idx_to_remove)
        new_gate = self._gate_factory.build_gate(removed_gate.qubits, self.use_evolutionary_strategy)
        target_col.add_gate(new_gate)
        circuit.columns[col_idx] = target_col
        return circuit


class ChangeDepthMutation(IMutationStrategy):
    def __init__(self, min_depth: int, max_depth: int, gate_factory: GateFactory, use_evolutionary_strategy: bool):
        # Define valores padrão seguros caso a configuração seja None
        self.min_depth = min_depth if min_depth is not None else 4
        self.max_depth = max_depth if max_depth is not None else 40
        self._gate_factory = gate_factory
        self.use_evolutionary_strategy = use_evolutionary_strategy

    def can_apply(self, individual: Circuit) -> bool:
        return True

    def mutate_individual(self, circuit: Circuit) -> Circuit:
        random_gauss = random.gauss(0, 1)
        change = math.ceil(random_gauss) if random.random() < 0.5 else math.floor(random_gauss)
        if change == 0: change = random.choice([-1, 1])

        new_depth = circuit.depth + change
        new_depth = max(self.min_depth, min(new_depth, self.max_depth))

        actual_change = new_depth - circuit.depth

        if actual_change < 0:
            for _ in range(abs(actual_change)):
                if circuit.columns:
                    circuit.columns.pop(random.randrange(len(circuit.columns)))
        elif actual_change > 0:
            for _ in range(actual_change):
                new_column = Column()
                qubits_free = list(range(circuit.count_qubits))
                while qubits_free:
                    try:
                        new_gate = self._gate_factory.build_gate(qubits_free, self.use_evolutionary_strategy)
                        new_column.add_gate(new_gate)
                        for q in new_gate.qubits:
                            qubits_free.remove(q)
                    except ValueError:
                        break
                circuit.columns.append(new_column)
        return circuit


class GateParameterMutation(IMutationStrategy):
    def __init__(self, fitness_evaluator: IFitnessEvaluator, c_factor: Optional[float] = None):
        self._fitness_evaluator = fitness_evaluator
        self._c_factor = c_factor

    def can_apply(self, circuit: Circuit) -> bool:
        return any(gate.parameters for col in circuit.columns for gate in col.get_gates())

    def mutate_individual(self, circuit: Circuit) -> Circuit:
        mutable_params = [
            (i_col, i_gate, i_param)
            for i_col, col in enumerate(circuit.columns)
            for i_gate, gate in enumerate(col.get_gates())
            for i_param in range(len(gate.parameters))
        ]
        if not mutable_params: return circuit

        i_col, i_gate, i_param = random.choice(mutable_params)
        target_gate = circuit.columns[i_col].gates[i_gate]
        
        if target_gate.steps_sizes and i_param < len(target_gate.steps_sizes):
            original_fitness, _ = self._fitness_evaluator.evaluate(circuit)
            step_size = target_gate.steps_sizes[i_param]
            change = random.gauss(0, step_size.sigma)
            target_gate.parameters[i_param] = (target_gate.parameters[i_param] + change) % (2 * math.pi)

            _reset_evaluation_flags(circuit)

            mutated_fitness, _ = self._fitness_evaluator.evaluate(circuit)
            
            success = mutated_fitness > original_fitness
            step_size.history.append(int(success))
            if len(step_size.history) > step_size.history_len:
                step_size.history.pop(0)

            success_rate = sum(step_size.history) / len(step_size.history)
            if success_rate > 1/5:
                step_size.sigma /= self._c_factor
            elif success_rate < 1/5:
                step_size.sigma *= self._c_factor
        else:
            change = random.gauss(0, math.pi / 4)
            target_gate.parameters[i_param] = (target_gate.parameters[i_param] + change) % (2 * math.pi)
            
        return circuit


class SwapControlTargetMutation(IMutationStrategy):
    CONTROL_GATE_NAMES = {"CXGate", "CZGate", "CYGate", "DCXGate", "ECRGate", "CHGate", "RCCXGate", "CU1Gate", "CU3Gate"}

    @staticmethod
    def _get_num_controls(gate_class):
        name = gate_class.__name__
        if name in {"CXGate", "CZGate", "CYGate", "DCXGate", "ECRGate", "CHGate", "CU1Gate", "CU3Gate"}: return 1
        if name == "RCCXGate": return 2
        return 0

    def can_apply(self, circuit: Circuit) -> bool:
        return any(self._get_num_controls(g.gate_class) > 0 for c in circuit.columns for g in c.get_gates())

    def mutate_individual(self, circuit: Circuit) -> Circuit:
        mutable_gates = [
            (i_col, i_gate, self._get_num_controls(gate.gate_class))
            for i_col, col in enumerate(circuit.columns)
            for i_gate, gate in enumerate(col.get_gates())
            if self._get_num_controls(gate.gate_class) > 0
        ]
        if not mutable_gates: return circuit

        i_col, i_gate, num_controls = random.choice(mutable_gates)
        target_gate = circuit.columns[i_col].gates[i_gate]
        
        control_qubits = target_gate.qubits[:num_controls]
        target_qubits = target_gate.qubits[num_controls:]
        
        if not control_qubits or not target_qubits: return circuit

        control_to_swap = random.choice(control_qubits)
        target_to_swap = random.choice(target_qubits)

        new_qubits = list(target_gate.qubits)
        idx_control = new_qubits.index(control_to_swap)
        idx_target = new_qubits.index(target_to_swap)
        new_qubits[idx_control], new_qubits[idx_target] = new_qubits[idx_target], new_qubits[idx_control]
        
        target_gate.qubits = new_qubits
        return circuit