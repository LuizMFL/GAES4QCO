import random
from inspect import Parameter, signature
from typing import Type, List, Dict, Tuple, Optional

from numpy import pi as PI
from qiskit.circuit import Gate as QiskitGate

from qiskit.circuit.library.standard_gates import (
    XGate, YGate, ZGate, HGate, SGate, TGate, SXGate,
    RXGate, RYGate, RZGate, PhaseGate, UGate, CXGate, CZGate,
    CYGate, DCXGate, ECRGate, RZXGate, RZZGate, RXXGate, RYYGate,
    CHGate, RCCXGate, IGate, RGate, CU1Gate, CU3Gate
)

from .gate import Gate
from shared.value_objects import StepSize


class GateFactory:
    """
    Cria uma entidade 'Gate' de forma aleatória,
    encapsulando as regras de seleção e parametrização.
    """
    def __init__(self, allowed_gates: Optional[List[str]] = None):
        """
        :param allowed_gates: Lista de nomes de gates permitidas.
        Se None, todas as gates do _gate_class_map serão usadas.
        """
        self._allowed_gates = set(allowed_gates) if allowed_gates else None

        # O mapa de gates é um detalhe de implementação da fábrica.
        gate_class_map: Dict[int, List[Type[QiskitGate]]] = {
            1: [XGate, YGate, ZGate, HGate, SGate, TGate, IGate, SXGate,
                UGate, PhaseGate, RGate, RXGate, RYGate, RZGate],
            2: [RXXGate, RYYGate, RZZGate, RZXGate,
                DCXGate, ECRGate, CXGate, CZGate, CYGate,
                CU1Gate, CU3Gate, CHGate],
            3: [RCCXGate],
        }

        # Filtra dinamicamente de acordo com allowed_gates
        self._gate_name_map: Dict[str, Type[QiskitGate]] = {
            cls.__name__: cls
            for _, gate_list in gate_class_map.items()
            for cls in gate_list
            if (self._allowed_gates is None or cls.__name__ in self._allowed_gates)
        }

        # Reconstrói o mapa filtrado
        self._gate_class_map: Dict[int, List[Type[QiskitGate]]] = {}
        for n, gate_list in gate_class_map.items():
            filtered = [cls for cls in gate_list if cls.__name__ in self._gate_name_map]
            if filtered:
                self._gate_class_map[n] = filtered

    def build_gate(self, available_qubits: List[int], use_evolutionary_strategy: bool) -> Gate:
        """
        Método principal para construir uma instância da nossa entidade Gate.
        """
        lim_qubits = len(available_qubits)
        if lim_qubits == 0:
            raise ValueError("Não há qubits disponíveis para criar um gate.")

        gate_class, qubits_min = self._choice_gate_class(lim_qubits)
        generated_angles = self._generate_random_params_for_gate(gate_class)
        steps_sizes = [StepSize() for _ in generated_angles] if use_evolutionary_strategy else None
        chosen_qubits = self._choice_qubits(available_qubits, qubits_min)

        # Retorna nossa entidade 'Gate', e não 'GateComponent'
        return Gate(
            gate_class=gate_class,
            qubits=chosen_qubits,
            parameters=generated_angles,
            steps_sizes=steps_sizes
        )

    def build_identity_gate(self, qubit: int) -> Gate:
        """
        Cria uma IGate (porta identidade) fixa em um qubit específico.
        Não adiciona aleatoriedade, usada em casos de preenchimento estrutural.
        """
        from qiskit.circuit.library.standard_gates import IGate

        return Gate(
            gate_class=IGate,
            qubits=[qubit],
            parameters=[],
            steps_sizes=None
        )

    def create_from_dict(self, data: dict) -> Gate:
        """
        Reconstrói uma entidade Gate a partir de um dicionário (proveniente de um JSON).
        """
        gate_name = data.get("gate_class_name")
        gate_class = self._gate_name_map.get(gate_name)

        if not gate_class:
            raise ValueError(f"Gate '{gate_name}' desconhecido. Não é possível reconstruir o circuito.")

        steps_sizes_dict = data.get("step_sizes")
        steps_sizes = [StepSize(**step_size) for step_size in steps_sizes_dict] if steps_sizes_dict else None

        return Gate(
            gate_class=gate_class,
            qubits=data.get("qubits"),
            parameters=data.get("parameters"),
            steps_sizes=steps_sizes
        )

    def _choice_gate_class(self, lim_qubits: int) -> Tuple[Type[QiskitGate], int]:
        candidate_pool = [
            (gate_class, num_qubits)
            for num_qubits, gate_list in self._gate_class_map.items()
            if num_qubits <= lim_qubits
            for gate_class in gate_list
        ]
        if not candidate_pool:
            raise ValueError(
                f"Não há gates disponíveis para {lim_qubits} qubits "
                f"com a configuração atual ({self._allowed_gates}"
            )
        return random.choice(candidate_pool)

    def _choice_qubits(self, qubits: List[int], min_qubits: int) -> List[int]:

        num_total_qubits = min_qubits
        chosen_qubits = random.sample(qubits, num_total_qubits)
        return chosen_qubits

    @classmethod
    def _generate_random_params_for_gate(cls, gate_class: Type[QiskitGate]) -> List[float]:
        """
        Usa introspecção para gerar ângulos aleatórios.
        Nota: Este método é poderoso, mas frágil. Mudanças na API do Qiskit
        podem exigir sua atualização.
        """
        sig = signature(gate_class.__init__)

        generated_angles = []
        for p in sig.parameters.values():
            # Verifica se é um parâmetro de ângulo que precisa ser preenchido
            if p.name not in ['self', 'label'] and p.default is Parameter.empty:
                angle = random.uniform(0, PI) if "theta" in p.name else random.uniform(0, 2 * PI)
                generated_angles.append(angle)
        return generated_angles
