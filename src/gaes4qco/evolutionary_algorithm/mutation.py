import random
import math
from typing import List, Optional

from qiskit.circuit import ControlledGate

from .interfaces import IMutationStrategy, IMutationPopulation
from .population import Population
from quantum_circuit.circuit import Circuit, Column
from quantum_circuit.gate_factory import GateFactory
from optimization.interfaces import IFitnessEvaluator


# --- Classe Composta para aplicar mutações aleatórias ---
class RandomMutationSelector(IMutationPopulation):
    def __init__(self, mutation_strategies: List[IMutationStrategy], mutation_rate: float = 0.1):
        self._strategies = mutation_strategies
        self.mutation_rate = mutation_rate

    def mutate(self, population: Population) -> Population:
        mutated_individuals = []
        for circuit in population.get_individuals():
            individual_copy = circuit.copy()
            if random.random() < self.mutation_rate:
                applicable_strategies = [s for s in self._strategies if s.can_apply(individual_copy)]
                if applicable_strategies:
                    strategy = random.choice(applicable_strategies)
                    mutated_circuit = strategy.mutate_individual(individual_copy)
                    mutated_individuals.append(mutated_circuit)
                else:
                    mutated_individuals.append(individual_copy)
            else:
                mutated_individuals.append(individual_copy)
        return Population(mutated_individuals)


# --- SELETOR ADAPTATIVO (MAB-UCB) ---
class BanditMutationSelector(IMutationPopulation):
    """
    Seleciona um operador de mutação usando o algoritmo Multi-Armed Bandit (UCB1).
    Implementação formal baseada em Auer et al. (2002).
    """

    def __init__(
        self,
        mutation_strategies: List[IMutationStrategy],
        mutation_rate: float,
        fitness_evaluator: IFitnessEvaluator
    ):
        self._strategies = mutation_strategies
        self.mutation_rate = mutation_rate
        self._fitness_evaluator = fitness_evaluator

        # Estruturas para MAB
        self._rewards = {s.__class__.__name__: 0.0 for s in self._strategies}
        self._counts = {s.__class__.__name__: 0 for s in self._strategies}
        self._total_applications = 0

    def _select_strategy(self, individual: Circuit) -> IMutationStrategy:
        """Seleciona a melhor estratégia usando UCB1 formal."""

        applicable_strategies = [s for s in self._strategies if s.can_apply(individual)]
        if not applicable_strategies:
            raise RuntimeError("Nenhuma estratégia aplicável disponível para o indivíduo.")

        # 1. Exploração inicial: use cada estratégia ao menos uma vez
        for s in applicable_strategies:
            if self._counts[s.__class__.__name__] == 0:
                return s

        # 2. Atualiza o número total de aplicações (n no UCB)
        n = self._total_applications if self._total_applications > 0 else 1

        # 3. Calcula UCB para cada estratégia
        ucb_scores = {}
        for s in applicable_strategies:
            name = s.__class__.__name__
            avg_reward = self._rewards[name] / self._counts[name]
            exploration_term = math.sqrt((2 * math.log(n)) / self._counts[name])
            ucb_scores[name] = avg_reward + exploration_term

        # 4. Seleciona a melhor estratégia
        best_strategy_name = max(ucb_scores, key=ucb_scores.get)
        return next(s for s in applicable_strategies if s.__class__.__name__ == best_strategy_name)

    def mutate(self, population: Population) -> Population:
        mutated_individuals = []
        for circuit in population.get_individuals():
            individual_copy = circuit.copy()
            if random.random() < self.mutation_rate:
                strategy = self._select_strategy(individual_copy)

                # Avaliação antes e depois da mutação
                original_fitness, _ = self._fitness_evaluator.evaluate(individual_copy)
                mutated_circuit = strategy.mutate_individual(individual_copy)
                mutated_fitness, _ = self._fitness_evaluator.evaluate(mutated_circuit)

                reward = mutated_fitness - original_fitness

                # Atualiza estatísticas MAB
                strategy_name = strategy.__class__.__name__
                self._counts[strategy_name] += 1
                self._rewards[strategy_name] += reward
                self._total_applications += 1

                mutated_individuals.append(mutated_circuit)
            else:
                mutated_individuals.append(individual_copy)

        return Population(mutated_individuals)



# --- Classes de Estratégia de Mutação Específicas ---

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
    """
    ## Altera a profundidade do circuito, adicionando ou removendo colunas.
    """

    def __init__(self, max_depth: int, gate_factory: GateFactory, use_evolutionary_strategy: bool):
        self.max_depth = max_depth
        self._gate_factory = gate_factory
        self.use_evolutionary_strategy = use_evolutionary_strategy

    def can_apply(self, individual: Circuit) -> bool:
        return True

    def mutate_individual(self, circuit: Circuit) -> Circuit:

        # Determina a mudança na profundidade com desvio padrão de 1
        random_gauss = random.gauss(0, 1)
        change = math.ceil(random_gauss) if random.random() < 0.5 else math.floor(random_gauss)
        if change == 0:
            change = random.choice([-1, 1])

        new_depth = circuit.depth + change
        # Garante que a profundidade fique entre [1, max_depth]
        new_depth = max(1, min(new_depth, self.max_depth))

        actual_change = new_depth - circuit.depth

        if actual_change < 0:  # Remove colunas
            for _ in range(abs(actual_change)):
                if circuit.columns:
                    col_to_remove = random.choice(circuit.columns)
                    circuit.columns.remove(col_to_remove)

        elif actual_change > 0:  # Adiciona colunas
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
                        break  # Não há mais gates que possam ser adicionados
                circuit.columns.append(new_column)

        return circuit


class GateParameterMutation(IMutationStrategy):
    """
    ## Ajusta um parâmetro de gate (ângulo) e atualiza o StepSize.
    ## Esta é a implementação da Estratégia Evolucionária.
    """

    def __init__(self, fitness_evaluator: IFitnessEvaluator, c_factor: Optional[float] = None):
        # ## Esta mutação PRECISA do avaliador de fitness para funcionar.
        # ## A injeção de dependência no construtor é a solução limpa.
        self._fitness_evaluator = fitness_evaluator
        self._c_factor = c_factor  # Fator de ajuste do StepSize

    def can_apply(self, circuit: Circuit) -> bool:
        # Aplicável se houver algum gate com parâmetros e step_sizes
        return any(
            gate.parameters
            for col in circuit.columns for gate in col.get_gates()
        )

    def mutate_individual(self, circuit: Circuit) -> Circuit:
        mutable_params = []
        for i_col, col in enumerate(circuit.columns):
            for i_gate, gate in enumerate(col.get_gates()):
                for i_param in range(len(gate.parameters)):
                    mutable_params.append((i_col, i_gate, i_param))
        original_fitness, _ = self._fitness_evaluator.evaluate(circuit)

        # Escolhe um parâmetro e o modifica
        i_col, i_gate, i_param = random.choice(mutable_params)
        target_gate = circuit.columns[i_col].gates[i_gate]
        if any(gate.steps_sizes for col in circuit.columns for gate in col.get_gates()):
            step_size = target_gate.steps_sizes[i_param]

            # Modifica o ângulo
            change = random.gauss(0, step_size.sigma)
            target_gate.parameters[i_param] = (target_gate.parameters[i_param] + change) % (2 * math.pi)
            circuit.columns[i_col].gates[i_gate] = target_gate
            # Avalia o fitness DEPOIS da mutação
            mutated_fitness, _ = self._fitness_evaluator.evaluate(circuit)

            # Regra de 1/5 de sucesso para atualizar o StepSize
            success = mutated_fitness > original_fitness
            step_size.history.append(int(success))
            if len(step_size.history) > step_size.history_len:
                step_size.history.pop(0)

            success_rate = sum(step_size.history) / len(step_size.history)
            if success_rate > 1 / 5:
                step_size.sigma /= self._c_factor
            elif success_rate < 1 / 5:
                step_size.sigma *= self._c_factor
        else:
            target_gate.parameters[i_param] = (target_gate.parameters[i_param] + random.gauss(0, math.pi / 4)) % (2 * math.pi)
            circuit.columns[i_col].gates[i_gate] = target_gate
        return circuit


class SwapControlTargetMutation(IMutationStrategy):
    """
    ## Em um gate controlado, troca um qubit de controle por um de alvo.

    Detecta automaticamente gates controlados com base em suas propriedades do Qiskit.
    """
    CONTROL_GATE_NAMES = {"CXGate", "CCXGate", "MCXGate", "RCCXGate", "RC3XGate", "MCX_GrayGate", "MCX_RecursiveGate"}

    @staticmethod
    def _get_num_controls(gate):
        # Permitir receber tanto a classe quanto a instância
        gate_name = getattr(gate, "__name__", gate.__class__.__name__).lower()
        # Se for um ControlledGate real (ex: CX, MCX, CRZ, CU3, etc.)
        if isinstance(gate, ControlledGate):
            return gate.num_ctrl_qubits
        # Fallback heurístico para gates compostos
        if gate_name in ("ccxgate", "rccxgate"):
            return 2
        if gate_name in ("rc3xgate", "mcxgate", "mcx_graygate", "mcx_recursivegate"):
            return 3
        return 0

    def can_apply(self, circuit: Circuit) -> bool:
        """Verifica se existe pelo menos um gate com qubits de controle."""

        return any(
            self._get_num_controls(gate.gate_class) > 0
            for column in circuit.columns
            for gate in column.get_gates()
        )

    def mutate_individual(self, circuit: Circuit) -> Circuit:
        """Realiza a mutação trocando um qubit de controle por um de alvo."""

        # Encontra todos os gates passíveis de mutação
        mutable_gates = []
        for i_col, col in enumerate(circuit.columns):
            for i_gate, gate in enumerate(col.get_gates()):
                num_controls = self._get_num_controls(gate.gate_class)
                if num_controls > 0:
                    mutable_gates.append((i_col, i_gate, num_controls))
        # Escolhe um gate aleatoriamente
        i_col, i_gate, num_controls = random.choice(mutable_gates)
        target_gate = circuit.columns[i_col].gates[i_gate]

        # Separa controles e alvos
        control_qubits = target_gate.qubits[:num_controls]
        target_qubits = target_gate.qubits[num_controls:]

        # Escolhe um controle e um alvo para trocar
        control_to_swap = random.choice(control_qubits)
        target_to_swap = random.choice(target_qubits)

        # Realiza a troca na lista de qubits
        new_qubits = list(target_gate.qubits)
        idx_control = new_qubits.index(control_to_swap)
        idx_target = new_qubits.index(target_to_swap)
        new_qubits[idx_control], new_qubits[idx_target] = new_qubits[idx_target], new_qubits[idx_control]

        # Atualiza o gate
        target_gate.qubits = new_qubits
        circuit.columns[i_col].gates[i_gate] = target_gate

        return circuit
