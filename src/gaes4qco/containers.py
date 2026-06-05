from dependency_injector import containers, providers
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit.providers.fake_provider.generic_backend_v2 import GenericBackendV2

from experiment import checkpoint, runner
from quantum_circuit import qiskit_adapter, circuit_factory, gate_factory, executor as quantum_executor
from evolutionary_algorithm import selection, crossover, mutation, population_factory, rate_adapter
from optimization import fitness, observer, optimizer, fitness_shaper
from analysis import error_analyzer, wrappers


class QuantumCircuitContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    qiskit_adapter = providers.Factory(qiskit_adapter.QiskitAdapter)
    gate_factory = providers.Factory(gate_factory.GateFactory, allowed_gates=config.quantum.allowed_gates)
    circuit_factory = providers.Factory(circuit_factory.CircuitFactory, gate_factory=gate_factory)


class OptimizationContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    gateways = providers.DependenciesContainer()

    target_statevector = providers.Singleton(Statevector, data=config.quantum.target_statevector_data)

    _evaluator = providers.Selector(
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
        )
    )
    evaluator = providers.Factory(wrappers.FitnessEvaluatorProfilerWrapper, decorated=_evaluator)

    _shaper = providers.Selector(
        config.selection_strategy.fitness_shaper,
        sharing=providers.Factory(
            fitness_shaper.FitnessSharingShaper,
            sharing_radius=config.niching.sharing_radius,
            alpha=config.niching.alpha
        ),
        default=providers.Factory(fitness_shaper.NullFitnessShaper)
    )
    shaper = providers.Factory(wrappers.FitnessShaperProfilerWrapper, decorated=_shaper)

    observer = providers.Factory(
        observer.JsonProgressObserver,
        filename=config.observer.filename,
        test_filename=config.observer.test_filename
    )


class EvolutionaryAlgorithmContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    factories = providers.DependenciesContainer()
    optimization = providers.DependenciesContainer()
    nsga2_service = providers.Factory(selection.NSGA2Service)

    _parent_selector = providers.Selector(
        config.selection_strategy.parent_selection,
        tournament=providers.Factory(selection.TournamentParentSelection, tournament_size=config.evolution.tournament_size),
        random=providers.Factory(selection.RandomParentSelection),
        roulette=providers.Factory(selection.RouletteParentSelection)
    )
    parent_selector = providers.Factory(wrappers.SelectionProfilerWrapper, decorated=_parent_selector)

    _survivor_selector = providers.Selector(
        config.selection_strategy.survivor_selection,
        tournament=providers.Factory(selection.TournamentSurvivorSelection, population_size=config.evolution.population_size, tournament_size=config.evolution.tournament_size, elitism_count=config.evolution.elitism_size),
        random=providers.Factory(selection.RandomSurvivorSelection, population_size=config.evolution.population_size, elitism_count=config.evolution.elitism_size),
        roulette=providers.Factory(selection.RouletteSurvivorSelection, population_size=config.evolution.population_size, elitism_count=config.evolution.elitism_size),
        nsga2=providers.Factory(selection.NSGA2SurvivorSelection, population_size=config.evolution.population_size, nsga2_service=nsga2_service, elitism_count=config.evolution.elitism_size)
    )
    survivor_selector = providers.Factory(wrappers.SelectionProfilerWrapper, decorated=_survivor_selector)

    _crossover_strategy_selector = providers.Selector(
        config.selection_strategy.crossover,
        singlepoint=providers.Factory(crossover.SinglePointCrossover),
        multipoint=providers.Factory(crossover.MultiPointCrossover),
        blockwise=providers.Factory(crossover.BlockwiseCrossover, gate_factory=factories.gate_factory)
    )
    
    _mutation_pool = providers.List(
        providers.Factory(mutation.SwapColumnsMutation),
        providers.Factory(mutation.SingleGateFlipMutation, gate_factory=factories.gate_factory, use_evolutionary_strategy=config.evolution.stepsize),
        providers.Factory(
            mutation.ChangeDepthMutation, 
            min_depth=config.evolution.min_depth,
            max_depth=config.evolution.max_depth, 
            gate_factory=factories.gate_factory, 
            use_evolutionary_strategy=config.evolution.stepsize
        ),
        providers.Factory(mutation.GateParameterMutation, fitness_evaluator=optimization.evaluator, c_factor=config.evolution.c_factor),
        providers.Factory(mutation.SwapControlTargetMutation)
    )
    
    mutation_selector = providers.Selector(
        config.selection_strategy.mutation,
        bandit=providers.Factory(mutation.BanditMutationSelector, mutation_strategies=_mutation_pool, mutation_rate=config.evolution.mutation_rate, fitness_evaluator=optimization.evaluator),
        default=providers.Factory(mutation.RandomMutationSelector, mutation_strategies=_mutation_pool, mutation_rate=config.evolution.mutation_rate),
    )
    
    _crossover_population = providers.Factory(
        crossover.PopulationCrossover,
        crossover_strategy=_crossover_strategy_selector,
        crossover_rate=config.evolution.crossover_rate
    )
    crossover_population = providers.Factory(wrappers.CrossoverProfilerWrapper, decorated=_crossover_population)

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
        )
    )


class AppContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    circuit = providers.Container(QuantumCircuitContainer, config=config)
    optimization = providers.Container(OptimizationContainer, config=config, gateways=circuit)
    evolutionary_algorithm = providers.Container(EvolutionaryAlgorithmContainer, config=config, factories=circuit, optimization=optimization)
    
    population_fac = providers.Factory(population_factory.PopulationFactory, circuit_factory=circuit.circuit_factory)
    
    checkpoint_manager = providers.Factory(
        checkpoint.CheckpointManager,
        config=config,
        population_factory=population_fac
    )

    optimizer = providers.Factory(
        optimizer.Optimizer,
        fitness_evaluator=optimization.evaluator,
        parent_selection=evolutionary_algorithm.parent_selector,
        survivor_selection=evolutionary_algorithm.survivor_selector,
        crossover=evolutionary_algorithm.crossover_population,
        mutation=evolutionary_algorithm.mutation_selector,
        rate_adapter=evolutionary_algorithm.rate_adapter,
        fitness_shaper=optimization.shaper,
        observer=optimization.observer
    )

    simulation_backend = providers.Factory(AerSimulator, method='statevector', device='GPU')
    
    quantum_executor = providers.Factory(quantum_executor.QiskitExecutor, adapter=circuit.qiskit_adapter, backend=simulation_backend)
    
    error_analyzer = providers.Factory(error_analyzer.ErrorAnalyzer, executor=quantum_executor)


class ExperimentContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    runner = providers.Factory(
        runner.ExperimentRunner,
        config=config,
        test_filename=config.test_filename,
        container=AppContainer
    )