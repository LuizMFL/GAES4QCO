from dependency_injector import containers, providers
from qiskit.quantum_info import Statevector
from qiskit_aer.noise import NoiseModel  # Importe o NoiseModel
from qiskit_aer import AerSimulator
from qiskit.providers.fake_provider.generic_backend_v2 import GenericBackendV2  # Exemplo de um backend simulado


from experiment import checkpoint, runner
from quantum_circuit import qiskit_adapter, circuit_factory, gate_factory, executor as quantum_executor
from evolutionary_algorithm import selection, crossover, mutation, population_factory, rate_adapter
from optimization import fitness, observer, optimizer, fitness_shaper
from analysis import error_analyzer


class QuantumCircuitContainer(containers.DeclarativeContainer):
    """Sub-container para os componentes da feature quantum_circuit."""
    config = providers.Configuration()

    qiskit_adapter = providers.Factory(qiskit_adapter.QiskitAdapter)
    gate_factory = providers.Factory(
        gate_factory.GateFactory,
        allowed_gates=config.quantum.allowed_gates
    )
    circuit_factory = providers.Factory(
        circuit_factory.CircuitFactory,
        gate_factory=gate_factory
    )


class OptimizationContainer(containers.DeclarativeContainer):
    """Sub-container para os componentes de avaliação e observação."""
    config = providers.Configuration()
    gateways = providers.DependenciesContainer()

    target_statevector = providers.Singleton(
        Statevector,
        data=config.quantum.target_statevector_data
    )

    # Seletor para a função de fitness
    evaluator = providers.Selector(
        config.selection_strategy.fitness,
        weighted=providers.Factory(
            fitness.WeightedFidelityFitnessEvaluator,
            target_statevector=target_statevector,
            circuit_adapter=gateways.qiskit_adapter,
            target_depth=config.quantum.target_depth
        ),
        default=providers.Factory(
            fitness.FidelityFitnessEvaluator,
            target_statevector=target_statevector,
            circuit_adapter=gateways.qiskit_adapter
        ),
    )

    shaper = providers.Selector(
        config.selection_strategy.fitness_shaper,
        sharing=providers.Factory(
            fitness_shaper.FitnessSharingShaper,
            sharing_radius=config.niching.sharing_radius,
            alpha=config.niching.alpha
        ),
        default=providers.Factory(
            fitness_shaper.NullFitnessShaper
        )
    )

    observer = providers.Factory(
        observer.JsonProgressObserver,
        filename=config.observer.filename
    )


class EvolutionaryAlgorithmContainer(containers.DeclarativeContainer):
    """Sub-container para as estratégias do algoritmo evolucionário."""
    config = providers.Configuration()
    factories = providers.DependenciesContainer()
    optimization = providers.DependenciesContainer()
    nsga2_service = providers.Factory(selection.NSGA2Service)

    # --- Estratégias de Seleção ---
    parent_selector = providers.Selector(
        config.selection_strategy.parent_selection,
        tournament=providers.Factory(
            selection.TournamentParentSelection,
            population_size=config.evolution.population_size,
            tournament_size=config.evolution.tournament_size
        ),
        random=providers.Factory(
            selection.RandomParentSelection,
            population_size=config.evolution.population_size
        ),
        roulette=providers.Factory(
            selection.RouletteParentSelection,
            population_size=config.evolution.population_size
        )
    )

    survivor_selector = providers.Selector(
        config.selection_strategy.survivor_selection,
        tournament=providers.Factory(
            selection.TournamentSurvivorSelection,
            population_size=config.evolution.population_size,
            tournament_size=config.evolution.tournament_size,
            elitism_count=config.evolution.elitism_size
        ),
        random=providers.Factory(
            selection.RandomSurvivorSelection,
            population_size=config.evolution.population_size,
            elitism_count=config.evolution.elitism_size
        ),
        roulette=providers.Factory(
            selection.RouletteSurvivorSelection,
            population_size=config.evolution.population_size,
            elitism_count=config.evolution.elitism_size
        ),
        nsga2=providers.Factory(
            selection.NSGA2SurvivorSelection,
            population_size=config.evolution.population_size,
            elitism_count=config.evolution.elitism_size,
            nsga2_service=nsga2_service
        )
    )

    singlepoint_crossover = providers.Factory(
        crossover.SinglePointCrossover
    )
    multipoint_crossover = providers.Factory(
        crossover.MultiPointCrossover
    )
    blockwise_crossover = providers.Factory(
        crossover.BlockwiseCrossover,
        gate_factory=factories.gate_factory
    )
    crossover_strategy_selector = providers.Selector(
        config.selection_strategy.crossover,
        singlepoint=singlepoint_crossover,
        multipoint=multipoint_crossover,
        blockwise=blockwise_crossover,
    )

    # --- Estratégia de Crossover ---
    crossover_population = providers.Factory(
        crossover.PopulationCrossover,
        crossover_strategy=crossover_strategy_selector,
        crossover_rate=config.evolution.crossover_rate
    )

    # --- Estratégias de Mutação ---
    mutation_pool = providers.List(
        providers.Factory(mutation.SwapColumnsMutation),
        providers.Factory(
            mutation.SingleGateFlipMutation,
            gate_factory=factories.gate_factory,
            use_evolutionary_strategy=config.evolution.stepsize
        ),
        providers.Factory(
            mutation.ChangeDepthMutation,
            max_depth=config.evolution.max_depth,
            gate_factory=factories.gate_factory,
            use_evolutionary_strategy=config.evolution.stepsize,
        ),
        providers.Factory(
            mutation.GateParameterMutation,
            fitness_evaluator=optimization.evaluator,
            c_factor=config.evolution.c_factor
        ),
        providers.Factory(mutation.SwapControlTargetMutation)
    )

    mutation_selector = providers.Selector(
        config.selection_strategy.mutation,
        bandit=providers.Factory(
            mutation.BanditMutationSelector,
            mutation_strategies=mutation_pool,
            mutation_rate=config.evolution.mutation_rate,
            fitness_evaluator=optimization.evaluator
        ),
        default=providers.Factory(
            mutation.RandomMutationSelector,
            mutation_strategies=mutation_pool,
            mutation_rate=config.evolution.mutation_rate
        ),
    )

    # --- Adaptador de Taxas ---
    rate_adapter = providers.Selector(
        config.selection_strategy.rate_adapter,
        adaptive=providers.Factory(
            rate_adapter.DiversityAdaptiveRateAdapter,
            min_mutation_rate=config.adaptive_rates.min_mutation_rate,
            max_mutation_rate=config.adaptive_rates.max_mutation_rate,
            min_crossover_rate=config.adaptive_rates.min_crossover_rate,
            max_crossover_rate=config.adaptive_rates.max_crossover_rate
        ),
        default=providers.Factory(
            rate_adapter.FixedRateAdapter,
            crossover_rate=config.evolution.crossover_rate,
            mutation_rate=config.evolution.mutation_rate
        ),
    )


class AppContainer(containers.DeclarativeContainer):
    """
    Container principal que agrega todos os sub-containers da aplicação.
    """
    config = providers.Configuration()

    # O CheckpointManager depende da configuração do experimento

    # --- Agregação dos Sub-Containers ---

    # 1. Componentes de Circuito Quântico (sem dependências externas)
    circuit = providers.Container(
        QuantumCircuitContainer,
        config=config
    )

    # 2. Componentes de Otimização (dependem do container de circuito)
    optimization = providers.Container(
        OptimizationContainer,
        config=config,
        gateways=circuit
    )

    # 3. Componentes do Algoritmo Evolucionário (dependem de factories e otimização)
    evolutionary_algorithm = providers.Container(
        EvolutionaryAlgorithmContainer,
        config=config,
        factories=circuit,
        optimization=optimization
    )
    population_fac = providers.Factory(
        population_factory.PopulationFactory,
        circuit_factory=circuit.circuit_factory
    )

    checkpoint_manager = providers.Factory(
        checkpoint.CheckpointManager,
        config=config.experiment,
        population_factory=population_fac
    )

    # 4. Montagem do Optimizer Principal (componente de mais alto nível)
    optimizer = providers.Factory(
        optimizer.Optimizer,
        fitness_evaluator=optimization.evaluator,
        parent_selection=evolutionary_algorithm.parent_selector,
        survivor_selection=evolutionary_algorithm.survivor_selector,
        crossover=evolutionary_algorithm.crossover_population,
        mutation=evolutionary_algorithm.mutation_selector,
        population_factory=population_fac,
        rate_adapter=evolutionary_algorithm.rate_adapter,
        diversity_threshold=config.evolution.diversity_threshold,
        injection_rate=config.evolution.injection_rate,
        fitness_shaper=optimization.shaper,
        observer=optimization.observer
    )

    noisy_backend = providers.Factory(
        AerSimulator
    )

    generic_backend = providers.Factory(
        GenericBackendV2,
        num_qubits=config.num_qubits,
    )

    noisy_backend_1 = providers.Factory(
        AerSimulator.from_backend,
        # Usamos um backend falso para obter um modelo de ruído realista
        backend=generic_backend
    )

    # Provider para o Executor
    quantum_executor = providers.Factory(
        quantum_executor.QiskitExecutor,
        adapter=circuit.qiskit_adapter,
        backend=noisy_backend
    )

    # Provider para o Analisador de Erro
    error_analyzer = providers.Factory(
        error_analyzer.ErrorAnalyzer,
        executor=quantum_executor
    )


class ExperimentContainer(containers.DeclarativeContainer):
    """Sub-container para os componentes da feature experiment."""
    config = providers.Configuration()

    # O ExperimentRunner depende do container principal e da configuração
    runner = providers.Factory(
        runner.ExperimentRunner,
        config=config,
        container=AppContainer
    )
