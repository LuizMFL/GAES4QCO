from qiskit.circuit import QuantumCircuit as QiskitCircuit
from .interfaces import IQuantumCircuitAdapter
from .circuit import Circuit
from .gate import Gate as DomainGate


class QiskitAdapter(IQuantumCircuitAdapter):
    """
    ## Implementa o Adapter para o backend Qiskit.
    ## Contém toda a lógica que depende diretamente da biblioteca Qiskit.
    """
    def from_domain(self, circuit: Circuit) -> QiskitCircuit:
        """
        ## Lógica do antigo método 'build' foi movida para cá.
        """
        qiskit_circuit = QiskitCircuit(circuit.count_qubits)

        for column in circuit.columns:
            for domain_gate in column.get_gates():
                # Constrói o gate do Qiskit a partir da nossa entidade
                gate_instance, qubits = self._build_gate_from_domain(domain_gate)
                qiskit_circuit.append(gate_instance, qubits)

        return qiskit_circuit

    def _build_gate_from_domain(self, domain_gate: DomainGate):
        """Helper para construir um único gate."""
        gate_instance = domain_gate.gate_class(*domain_gate.parameters)
        return gate_instance, domain_gate.qubits
