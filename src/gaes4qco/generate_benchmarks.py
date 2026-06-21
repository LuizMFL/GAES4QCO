import json
import numpy as np
from pathlib import Path
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT

# Mapeamento estrito do Qiskit para as classes de domínio do seu GA
GATE_MAP = {
    'h': 'HGate',
    'cx': 'CXGate',
    'rx': 'RXGate',
    'ry': 'RYGate',
    'rz': 'RZGate',
    'id': 'IGate'
}


def qiskit_to_domain_json(qc: QuantumCircuit, num_qubits: int = 4) -> dict:
    """
    Converte um QuantumCircuit para o formato JSON do GAES4QCO.
    Empacota as portas nas colunas mais curtas possíveis (ASAP Scheduling)
    para calcular a profundidade lógica real.
    """
    columns = []
    # Rastreador para saber a última coluna ocupada por cada qubit
    qubit_to_col = {q: -1 for q in range(num_qubits)}

    for instruction in qc.data:
        qargs = [qc.find_bit(q).index for q in instruction.qubits]

        # Descobre a coluna mais à esquerda onde esta porta pode entrar
        target_col = max(qubit_to_col[q] for q in qargs) + 1

        # Cria colunas vazias se necessário
        while len(columns) <= target_col:
            columns.append({"gates": []})

        # Atualiza o rastreador
        for q in qargs:
            qubit_to_col[q] = target_col

        gate_name = instruction.operation.name
        if gate_name not in GATE_MAP:
            # Caso o transpiler vaze alguma porta extra, estouramos erro
            raise ValueError(f"Porta '{gate_name}' não mapeada na sua arquitetura!")

        params = [float(p) for p in instruction.operation.params]

        columns[target_col]["gates"].append({
            "gate_class_name": GATE_MAP[gate_name],
            "qubits": qargs,
            "parameters": params,
            "step_sizes": []
        })

    return {
        "count_qubits": num_qubits,
        "depth": len(columns),
        "fitness": None,
        "base_fitness": None,
        "fidelity": 1.0,
        "nsga2_rank": -1,
        "nsga2_crowding_distance": 0.0,
        "columns": columns
    }


def main():
    print("🧠 Construindo estados teóricos perfeitos...")
    num_qubits = 4

    # 1. ESTADO GHZ (O Teste de Emaranhamento)
    # Exige uma cascata perfeita de CNOTs. Otimizadores ruins adicionam ruído aqui.
    qc_ghz = QuantumCircuit(num_qubits)
    qc_ghz.h(0)
    qc_ghz.cx(0, 1)
    qc_ghz.cx(1, 2)
    qc_ghz.cx(2, 3)

    # 2. QFT - Transformada Quântica de Fourier (O Teste de Roteamento)
    # Altamente conectada. O transpiler da IBM insere muitos SWAPs aqui.
    # Veremos se a Fase 2 do seu GA acha uma topologia mais limpa.
    qc_qft = QFT(num_qubits).decompose()

    # 3. ESTADO W (O Teste de Sintonia Fina)
    # A distribuição é de 1/4 para cada estado (1000, 0100, 0010, 0001).
    # Exige rotações não inteiras perfeitas. A Fase 3 (StepSize) precisa brilhar aqui.
    w_array = np.zeros(2 ** num_qubits)
    w_array[1] = w_array[2] = w_array[4] = w_array[8] = 0.5
    qc_w = QuantumCircuit(num_qubits)
    qc_w.prepare_state(w_array)

    # Transpila todos para a base rigorosa de 6 portas do seu GA
    print("⚙️ Transpilando para o domínio [I, H, RX, RY, RZ, CX]...")
    basis_gates = ['id', 'h', 'rx', 'ry', 'rz', 'cx']

    qc_ghz_t = transpile(qc_ghz, basis_gates=basis_gates, optimization_level=3)
    qc_qft_t = transpile(qc_qft, basis_gates=basis_gates, optimization_level=3)
    qc_w_t = transpile(qc_w, basis_gates=basis_gates, optimization_level=3)

    # Exporta para os diretórios alvo
    out_dir = Path("results/target_circuits")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "target_GHZ.json", "w") as f:
        json.dump(qiskit_to_domain_json(qc_ghz_t), f, indent=4)

    with open(out_dir / "target_QFT.json", "w") as f:
        json.dump(qiskit_to_domain_json(qc_qft_t), f, indent=4)

    with open(out_dir / "target_W_STATE.json", "w") as f:
        json.dump(qiskit_to_domain_json(qc_w_t), f, indent=4)

    print(f"✅ Arquivos criados com sucesso em {out_dir}/")
    print("   - target_GHZ.json")
    print("   - target_QFT.json")
    print("   - target_W_STATE.json")


if __name__ == "__main__":
    main()
