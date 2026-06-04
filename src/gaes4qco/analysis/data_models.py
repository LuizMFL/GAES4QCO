from dataclasses import dataclass, field
from typing import List


@dataclass
class ResultData:
    """
    Contém os dados de resultado de uma única execução do algoritmo.
    Armazena as séries temporais completas para análise.
    """
    best_fitness_per_generation: List[float] = field(default_factory=list)
    average_fitness_per_generation: List[float] = field(default_factory=list)
    std_dev_fitness_per_generation: List[float] = field(default_factory=list)
    
    best_fidelity_per_generation: List[float] = field(default_factory=list)
    average_fidelity_per_generation: List[float] = field(default_factory=list)
    std_dev_fidelity_per_generation: List[float] = field(default_factory=list)
    
    best_depth_per_generation: List[float] = field(default_factory=list)
    average_depth_per_generation: List[float] = field(default_factory=list)
    std_dev_depth_per_generation: List[float] = field(default_factory=list)
    
    structural_diversity_per_generation: List[float] = field(default_factory=list)

    @property
    def generation_count(self) -> int:
        """Retorna o número de gerações registradas."""
        return len(self.best_fitness_per_generation)

    @property
    def max_depth_per_generation(self) -> List[float]:
        """Propriedade para manter a compatibilidade com o plotter."""
        # Em uma análise mais detalhada, a "melhor" profundidade pode ser a menor.
        # No entanto, o plotter usa "best_depth" como a profundidade do melhor indivíduo.
        return self.best_depth_per_generation
