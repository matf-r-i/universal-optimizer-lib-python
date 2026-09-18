"""Bacterial Foraging Optimization and its construction parameters"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from uo.algorithm.metaheuristic.additional_statistics_control import (
    AdditionalStatisticsControl,
)
from uo.algorithm.metaheuristic.bacterial_foraging.bfo_movement_support import (
    BfoMovementSupport,
)
from uo.algorithm.metaheuristic.bacterial_foraging.bfo_step_size_support import (
    BfoStepSizeSupport,
)
from uo.algorithm.metaheuristic.bacterial_foraging.bfo_swarming_support import (
    BfoSwarmingSupport,
)
from uo.algorithm.metaheuristic.finish_control import FinishControl
from uo.algorithm.metaheuristic.population_based_metaheuristic import (
    PopulationBasedMetaheuristic,
)
from uo.algorithm.output_control import OutputControl
from uo.problem.problem import Problem
from uo.solution.solution import Solution


@dataclass
class BfoOptimizerConstructionParameters:
    """Collect all arguments accepted by :class:`BfoOptimizer`

    :class:`BfoOptimizer` validates that all required values are present when it is constructed
    """

    movement_support: BfoMovementSupport | None = None
    swarming_support: BfoSwarmingSupport | None = None
    step_size_support: BfoStepSizeSupport | None = None
    population_size: int | None = None
    chemotactic_steps: int | None = None
    swim_length: int | None = None
    reproduction_steps: int | None = None
    elimination_dispersal_events: int | None = None
    elimination_dispersal_probability: float | None = None
    finish_control: FinishControl | None = None
    problem: Problem | None = None
    solution_template: Solution | None = None
    output_control: OutputControl | None = None
    random_seed: int | None = None
    additional_statistics_control: AdditionalStatisticsControl | None = None


class BfoOptimizer(PopulationBasedMetaheuristic):
    """Coordinate Bacterial Foraging Optimization phases and strategies
    """

    def __init__(
        self,
        movement_support: BfoMovementSupport,
        swarming_support: BfoSwarmingSupport,
        step_size_support: BfoStepSizeSupport,
        population_size: int,
        chemotactic_steps: int,
        swim_length: int,
        reproduction_steps: int,
        elimination_dispersal_events: int,
        elimination_dispersal_probability: float,
        finish_control: FinishControl,
        problem: Problem,
        solution_template: Solution,
        output_control: OutputControl | None = None,
        random_seed: int | None = None,
        additional_statistics_control: AdditionalStatisticsControl | None = None,
    ) -> None:
        """Create a BFO optimizer

        :param BfoMovementSupport movement_support: representation-specific
            tumble and movement strategy
        :param BfoSwarmingSupport swarming_support: social-interaction strategy
        :param BfoStepSizeSupport step_size_support: step initialization and
            adaptation strategy
        :param int population_size: even number of bacteria, at least two
        :param int chemotactic_steps: chemotactic sweeps before reproduction
        :param int swim_length: maximum accepted swim moves after a tumble
        :param int reproduction_steps: reproductions before elimination
        :param int elimination_dispersal_events: elimination-dispersal events
            before natural termination
        :param float elimination_dispersal_probability: independent replacement
            probability in the interval ``[0, 1]``
        :param FinishControl finish_control: framework termination criteria
        :param Problem problem: single-objective problem being optimized
        :param Solution solution_template: template used to create bacteria
        :param OutputControl | None output_control: optional output settings
        :param int | None random_seed: seed for deterministic randomness
        :param AdditionalStatisticsControl | None additional_statistics_control:
            optional statistics settings
        """
        self._validate_strategy(
            "movement_support", movement_support, BfoMovementSupport
        )
        self._validate_strategy(
            "swarming_support", swarming_support, BfoSwarmingSupport
        )
        self._validate_strategy(
            "step_size_support", step_size_support, BfoStepSizeSupport
        )
        self._validate_integer("population_size", population_size, minimum=2)
        if population_size % 2 != 0:
            raise ValueError("Parameter 'population_size' must be even")
        self._validate_integer("chemotactic_steps", chemotactic_steps, minimum=1)
        self._validate_integer("swim_length", swim_length, minimum=0)
        self._validate_integer("reproduction_steps", reproduction_steps, minimum=1)
        self._validate_integer(
            "elimination_dispersal_events",
            elimination_dispersal_events,
            minimum=1,
        )
        if isinstance(elimination_dispersal_probability, bool) or not isinstance(
            elimination_dispersal_probability, (int, float)
        ):
            raise TypeError(
                "Parameter 'elimination_dispersal_probability' must be "
                "'float' or 'int'"
            )
        if not 0.0 <= elimination_dispersal_probability <= 1.0:
            raise ValueError(
                "Parameter 'elimination_dispersal_probability' must be in [0, 1]"
            )
        if not isinstance(problem, Problem):
            raise TypeError("Parameter 'problem' must be 'Problem'")
        if problem.is_multi_objective:
            raise ValueError("BFO supports only single-objective problems")
        if solution_template is None:
            raise ValueError("Parameter 'solution_template' must not be None")

        super().__init__(
            finish_control=finish_control,
            problem=problem,
            solution_template=solution_template,
            name="bfo",
            output_control=output_control,
            random_seed=random_seed,
            additional_statistics_control=additional_statistics_control,
        )

        self.__movement_support = movement_support
        self.__swarming_support = swarming_support
        self.__step_size_support = step_size_support
        self.__population_size = population_size
        self.__chemotactic_steps = chemotactic_steps
        self.__swim_length = swim_length
        self.__reproduction_steps = reproduction_steps
        self.__elimination_dispersal_events = elimination_dispersal_events
        self.__elimination_dispersal_probability = float(
            elimination_dispersal_probability
        )

        self.__current_population: list[Solution] = []
        self.__health: list[float] = []
        self.__step_sizes: list[float] = []
        self.__chemotactic_step = 0
        self.__reproduction_step = 0
        self.__elimination_dispersal_step = 0
        self.__random_generator = Random(self.random_seed)

    @staticmethod
    def _validate_strategy(name: str, value: object, expected_type: type) -> None:
        """Validate one injected BFO strategy"""
        if not isinstance(value, expected_type):
            raise TypeError(
                f"Parameter '{name}' must be '{expected_type.__name__}'"
            )

    @staticmethod
    def _validate_integer(name: str, value: object, minimum: int) -> None:
        """Validate an integer value and its inclusive minimum"""
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"Parameter '{name}' must be 'int'")
        if value < minimum:
            qualifier = "non-negative" if minimum == 0 else f"at least {minimum}"
            raise ValueError(f"Parameter '{name}' must be {qualifier}")

    @classmethod
    def from_construction_tuple(
        cls,
        construction_tuple: BfoOptimizerConstructionParameters,
    ) -> BfoOptimizer:
        """Construct an optimizer from grouped construction parameters

        :param BfoOptimizerConstructionParameters construction_tuple: grouped
            optimizer arguments
        :return: configured BFO optimizer
        :rtype: BfoOptimizer
        """
        if not isinstance(construction_tuple, BfoOptimizerConstructionParameters):
            raise TypeError(
                "Parameter 'construction_tuple' must be "
                "'BfoOptimizerConstructionParameters'"
            )
        return cls(
            movement_support=construction_tuple.movement_support,
            swarming_support=construction_tuple.swarming_support,
            step_size_support=construction_tuple.step_size_support,
            population_size=construction_tuple.population_size,
            chemotactic_steps=construction_tuple.chemotactic_steps,
            swim_length=construction_tuple.swim_length,
            reproduction_steps=construction_tuple.reproduction_steps,
            elimination_dispersal_events=(
                construction_tuple.elimination_dispersal_events
            ),
            elimination_dispersal_probability=(
                construction_tuple.elimination_dispersal_probability
            ),
            finish_control=construction_tuple.finish_control,
            problem=construction_tuple.problem,
            solution_template=construction_tuple.solution_template,
            output_control=construction_tuple.output_control,
            random_seed=construction_tuple.random_seed,
            additional_statistics_control=(
                construction_tuple.additional_statistics_control
            ),
        )

    def copy(self) -> BfoOptimizer:
        """Return a copy of this optimizer and its mutable runtime state

        :return: copied BFO optimizer
        :rtype: BfoOptimizer
        """
        copied = BfoOptimizer(
            movement_support=self.movement_support.copy(),
            swarming_support=self.swarming_support.copy(),
            step_size_support=self.step_size_support.copy(),
            population_size=self.population_size,
            chemotactic_steps=self.chemotactic_steps,
            swim_length=self.swim_length,
            reproduction_steps=self.reproduction_steps,
            elimination_dispersal_events=self.elimination_dispersal_events,
            elimination_dispersal_probability=(
                self.elimination_dispersal_probability
            ),
            finish_control=self.finish_control.copy(),
            problem=self.problem.copy(),
            solution_template=self.solution_template.copy(),
            output_control=(
                self.output_control.copy() if self.output_control is not None else None
            ),
            random_seed=self.random_seed,
            additional_statistics_control=self.additional_statistics_control,
        )
        copied.__current_population = [
            bacterium.copy() for bacterium in self.current_population
        ]
        copied.__health = self.health.copy()
        copied.__step_sizes = self.step_sizes.copy()
        copied.__chemotactic_step = self.chemotactic_step
        copied.__reproduction_step = self.reproduction_step
        copied.__elimination_dispersal_step = self.elimination_dispersal_step
        copied.evaluation = self.evaluation
        copied.iteration = self.iteration
        self._copy_runtime_state_to(copied)
        copied.__random_generator.setstate(self.__random_generator.getstate())
        if self.best_solution is not None:
            copied.evaluation_best_found = self.evaluation_best_found
            copied.iteration_best_found = self.iteration_best_found
        return copied

    @property
    def movement_support(self) -> BfoMovementSupport:
        """Return the representation-specific movement strategy"""
        return self.__movement_support

    @property
    def swarming_support(self) -> BfoSwarmingSupport:
        """Return the social-interaction strategy"""
        return self.__swarming_support

    @property
    def step_size_support(self) -> BfoStepSizeSupport:
        """Return the chemotactic step-size strategy"""
        return self.__step_size_support

    @property
    def population_size(self) -> int:
        """Return the configured number of bacteria"""
        return self.__population_size

    @property
    def chemotactic_steps(self) -> int:
        """Return the number of sweeps before reproduction"""
        return self.__chemotactic_steps

    @property
    def swim_length(self) -> int:
        """Return the maximum number of accepted swim moves per tumble"""
        return self.__swim_length

    @property
    def reproduction_steps(self) -> int:
        """Return the number of reproductions before elimination-dispersal"""
        return self.__reproduction_steps

    @property
    def elimination_dispersal_events(self) -> int:
        """Return the configured number of elimination-dispersal events"""
        return self.__elimination_dispersal_events

    @property
    def elimination_dispersal_probability(self) -> float:
        """Return the independent probability of dispersing a bacterium"""
        return self.__elimination_dispersal_probability

    @property
    def current_population(self) -> list[Solution]:
        """Return the live bacterial population"""
        return self.__current_population

    @current_population.setter
    def current_population(self, value: list[Solution]) -> None:
        """Replace the live population after validating its contents

        :param list[Solution] value: new bacterial population
        """
        if not isinstance(value, list):
            raise TypeError("Parameter 'current_population' must have type 'list'")
        if any(not isinstance(bacterium, Solution) for bacterium in value):
            raise TypeError(
                "Every item in 'current_population' must have type 'Solution'"
            )
        self.__current_population = value
        self.__population_size = len(value)

    @property
    def health(self) -> list[float]:
        """Return accumulated effective fitness for the live bacteria"""
        return self.__health

    @property
    def step_sizes(self) -> list[float]:
        """Return the current per-bacterium chemotactic step sizes"""
        return self.__step_sizes

    @property
    def chemotactic_step(self) -> int:
        """Return the current chemotactic-sweep index"""
        return self.__chemotactic_step

    @property
    def reproduction_step(self) -> int:
        """Return the current reproduction index"""
        return self.__reproduction_step

    @property
    def elimination_dispersal_step(self) -> int:
        """Return the number of completed elimination-dispersal events"""
        return self.__elimination_dispersal_step

    def init(self) -> None:
        """Initialize BFO runtime state 
        """
        self.__random_generator = Random(self.random_seed)
        self.evaluation = 0
        self.iteration = 0
        self.__chemotactic_step = 0
        self.__reproduction_step = 0
        self.__elimination_dispersal_step = 0
        self.__current_population = []
        self.__health = []
        self.__step_sizes = []

        for _ in range(self.population_size):
            bacterium = self.solution_template.copy()
            bacterium.init_random(self.problem)
            bacterium.evaluate(self.problem)
            self.evaluation += 1
            self.__current_population.append(bacterium)
            self.__health.append(0.0)
            self.__step_sizes.append(self._validated_step_size(
                self.step_size_support.initial_step_size()
            ))

        self.best_solution = max(
            self.current_population,
            key=lambda bacterium: bacterium.fitness_value,
        )
        self.update_additional_statistics_if_required(self.best_solution)

    def should_finish(self) -> bool:
        """Return whether BFO reached an external or natural stopping condition"""
        return (
            self.elimination_dispersal_step >= self.elimination_dispersal_events
            or super().should_finish()
        )

    def main_loop_iteration(self) -> None:
        """Execute one chemotactic sweep 
        """
        for index, bacterium in enumerate(self.__current_population):
            direction = self.movement_support.tumble_direction(
                bacterium,
                self.__random_generator,
            )
            current = bacterium
            current_effective_fitness = self._effective_fitness(current)
            health = 0.0

            for _ in range(self.swim_length + 1):
                if self._evaluation_limit_reached():
                    return
                candidate = self.movement_support.move(
                    current,
                    direction,
                    self.__step_sizes[index],
                    self.problem,
                )
                if not isinstance(candidate, Solution):
                    raise TypeError("BFO movement support must return a Solution")
                candidate.evaluate(self.problem)
                self.evaluation += 1
                candidate_effective_fitness = self._effective_fitness(candidate)
                improved = candidate_effective_fitness > current_effective_fitness
                self.__step_sizes[index] = self._validated_step_size(
                    self.step_size_support.adapt(
                        self.__step_sizes[index],
                        improved,
                    )
                )

                if not improved:
                    health += current_effective_fitness
                    break

                current = candidate
                current_effective_fitness = candidate_effective_fitness
                health += current_effective_fitness
                self.__current_population[index] = current
                self._update_best_solution(current)

            self.__health[index] = health

        self.__chemotactic_step += 1
        self.iteration += 1

        if self.__chemotactic_step < self.chemotactic_steps:
            return

        self.__chemotactic_step = 0
        self.__reproduction_step += 1
        if self.__reproduction_step < self.reproduction_steps:
            return

        self.__reproduction_step = 0
        self._reproduce()
        self._eliminate_and_disperse()
        self.__elimination_dispersal_step += 1

    def _effective_fitness(self, bacterium: Solution) -> float:
        interaction = self.swarming_support.interaction_value(
            bacterium,
            self.__current_population,
        )
        if isinstance(interaction, bool) or not isinstance(interaction, (int, float)):
            raise TypeError("BFO swarming support must return a number")
        return float(bacterium.fitness_value) + float(interaction)

    def _update_best_solution(self, candidate: Solution) -> None:
        if self.best_solution is None or candidate.is_better(self.best_solution, self.problem):
            self.best_solution = candidate
            self.update_additional_statistics_if_required(candidate)

    def _reproduce(self) -> None:
        ranked_indices = sorted(
            range(self.population_size),
            key=lambda index: self.__health[index],
            reverse=True,
        )
        half = self.population_size // 2
        for position in range(half, self.population_size):
            source = ranked_indices[position - half]
            replacement = self.__current_population[source].copy()
            self.__current_population[position] = replacement
            self.__health[position] = self.__health[source]
            self.__step_sizes[position] = self.__step_sizes[source]

    def _eliminate_and_disperse(self) -> None:
        for index, bacterium in enumerate(self.__current_population):
            if self._evaluation_limit_reached():
                return
            if self.__random_generator.random() >= self.elimination_dispersal_probability:
                continue
            replacement = self.solution_template.copy()
            replacement.init_random(self.problem)
            replacement.evaluate(self.problem)
            self.evaluation += 1
            self.__current_population[index] = replacement
            self.__health[index] = 0.0
            self.__step_sizes[index] = self._validated_step_size(
                self.step_size_support.initial_step_size()
            )
            self._update_best_solution(replacement)

    def _evaluation_limit_reached(self) -> bool:
        return (
            self.finish_control.check_evaluations
            and self.evaluation >= self.finish_control.evaluations_max
        )

    @staticmethod
    def _validated_step_size(value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("BFO step-size support must return a number")
        if value <= 0:
            raise ValueError("BFO step-size support must return a positive value")
        return float(value)

    def string_rep(
        self,
        delimiter: str,
        indentation: int = 0,
        indentation_symbol: str = "",
        group_start: str = "{",
        group_end: str = "}",
    ) -> str:
        """Return a configurable string representation of the optimizer

        :param str delimiter: delimiter between fields
        :param int indentation: indentation level
        :param str indentation_symbol: repeated indentation symbol
        :param str group_start: opening group marker
        :param str group_end: closing group marker
        :return: optimizer representation
        :rtype: str
        """
        inherited = super().string_rep(
            delimiter, indentation, indentation_symbol, "", ""
        )
        indent = indentation_symbol * indentation
        fields = [
            f"movement_support={type(self.movement_support).__name__}",
            f"swarming_support={type(self.swarming_support).__name__}",
            f"step_size_support={type(self.step_size_support).__name__}",
            f"population_size={self.population_size}",
            f"chemotactic_steps={self.chemotactic_steps}",
            f"swim_length={self.swim_length}",
            f"reproduction_steps={self.reproduction_steps}",
            f"elimination_dispersal_events={self.elimination_dispersal_events}",
            (
                "elimination_dispersal_probability="
                f"{self.elimination_dispersal_probability}"
            ),
            f"current_population={self.current_population}",
            f"health={self.health}",
            f"step_sizes={self.step_sizes}",
            f"chemotactic_step={self.chemotactic_step}",
            f"reproduction_step={self.reproduction_step}",
            f"elimination_dispersal_step={self.elimination_dispersal_step}",
        ]
        suffix = delimiter.join(indent + field for field in fields)
        return (
            delimiter
            + indent
            + group_start
            + inherited
            + delimiter
            + suffix
            + delimiter
            + indent
            + group_end
        )

    def __str__(self) -> str:
        """Return the compact string representation"""
        return self.string_rep("|")

    def __repr__(self) -> str:
        """Return the multiline string representation"""
        return self.string_rep("\n")

    def __format__(self, spec: str) -> str:
        """Return the conventional formatted representation

        :param str spec: accepted for Python formatting compatibility
        :return: formatted optimizer representation
        :rtype: str
        """
        return self.string_rep("\n", 0, "   ", "{", "}")
