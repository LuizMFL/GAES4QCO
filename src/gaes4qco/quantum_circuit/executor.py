# src/gaes4qco/quantum_circuit/executor.py (Novo Arquivo)

from typing import Dict, Union
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit.providers.backend import Backend

from .interfaces import IQuantumCircuitAdapter, IQuantumExecutor
from .circuit import Circuit


class QiskitExecutor(IQuantumExecutor):
    """
    Executa um circuito de domínio em um backend Qiskit (real ou simulado),
    adicionando medições e retornando as contagens.
    """

    def __init__(self, adapter: IQuantumCircuitAdapter, backend: Backend):
        self._adapter = adapter
        self._backend = backend

    def execute(self, circuit: Circuit, shots: int, measure: bool = True) -> Union[Dict[str, int], Statevector]:
        # Converte o circuito de domínio para Qiskit
        qiskit_circuit = self._adapter.from_domain(circuit)
        transpiled_circuit = transpile(qiskit_circuit, self._backend)
        print(f"Transpiled Circuit Depth: {transpiled_circuit.depth()}")
        # Se for medir (modo ruidoso)
        if measure:
            transpiled_circuit.measure_all()
            job = self._backend.run(transpiled_circuit, shots=shots)
            result = job.result()
            counts = result.get_counts(transpiled_circuit)
            return counts

        # Se não for medir, tenta gerar statevector determinístico
        # Usamos sempre Statevector.from_instruction, que funciona mesmo com gates não nativos
        sv = Statevector.from_instruction(transpiled_circuit)
        return sv
