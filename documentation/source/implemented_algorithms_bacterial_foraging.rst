.. _Algorithm_Bacterial_Foraging_Optimization:

Bacterial Foraging Optimization
================================

Basic information
---------------------

Bacterial Foraging Optimization (BFO) is a population-based metaheuristic for
single-objective continuous optimization. It models a group of bacteria that
move through the search space. A bacterium tumbles in a random direction and
continues swimming in that direction while the effective fitness improves.
After a number of `chemotactic sweeps` (tumbling of bacteria, in biology due to chemical signal),
the population is reproduced and some
bacteria may be eliminated and dispersed to new random positions.

The optimizer follows the library convention that larger fitness values are
better. Social interaction contributes to a temporary effective fitness used
for movement and health calculations, it does not change a solution's stored
objective or fitness value. The optimizer keeps the best evaluated solution
separately from the live population, so dispersal cannot remove the result
already found.

Implementation notes
---------------------

The main implementation is
:class:`~uo.algorithm.metaheuristic.bacterial_foraging.BfoOptimizer`.
Movement, social interaction, and step-size handling are supplied as separate
strategies:

* :class:`~uo.algorithm.metaheuristic.bacterial_foraging.BfoMovementSupport`
  creates tumble directions and moved candidates. The built-in real-valued
  implementation is
  :class:`~uo.algorithm.metaheuristic.bacterial_foraging.BfoMovementSupportReal`.
* :class:`~uo.algorithm.metaheuristic.bacterial_foraging.BfoSwarmingSupportIdle`
  disables social interaction. For distance-based attraction and repulsion,
  use
  :class:`~uo.algorithm.metaheuristic.bacterial_foraging.BfoSwarmingSupportReal`.
* :class:`~uo.algorithm.metaheuristic.bacterial_foraging.BfoStepSizeSupportFixed`
  keeps one step size. The adaptive alternative is
  :class:`~uo.algorithm.metaheuristic.bacterial_foraging.BfoStepSizeSupportAdaptive`.

The package also exports the abstract support classes and
:class:`~uo.algorithm.metaheuristic.bacterial_foraging.BfoOptimizerConstructionParameters`.

Algorithm steps
---------------------

.. rubric:: Chemotaxis and swimming

One optimizer iteration performs one chemotactic sweep over the population.
For each bacterium, BFO generates a tumble direction and evaluates a moved
candidate. If the candidate improves the effective fitness, it becomes the
current bacterium and the same direction is used for up to ``swim_length``
more moves. A rejected move ends that bacterium's swim. Every candidate
evaluation increments the optimizer's ``evaluation`` counter.

.. rubric:: Health and effective fitness

For one chemotactic sweep, a bacterium's health is the sum of the effective
fitness values encountered during its movement. With idle swarming, effective
fitness is the solution's fitness. With real swarming, the interaction value
is added temporarily for movement decisions and health calculation. It is not
written back to the solution.

.. rubric:: Reproduction

After ``chemotactic_steps`` sweeps, bacteria are ranked by health. The
healthiest half is retained, and each retained bacterium is copied once to
restore the configured population size. Health and step sizes are copied with
the parent.

.. rubric:: Elimination and dispersal

After ``reproduction_steps`` reproduction events, each bacterium is considered
independently. With probability ``elimination_dispersal_probability``, it is
replaced by a newly randomized and evaluated bacterium. The replacement starts
with zero health and the configured initial step size. The separate global
best is kept when a live bacterium is dispersed.

.. rubric:: Parameters

``BfoOptimizer`` accepts the following arguments:

* ``movement_support``: movement strategy for the solution representation.
  The built-in real strategy expects a ``list[float]`` or
  ``tuple[float, ...]`` and inclusive lower and upper bounds for each
  dimension.
* ``swarming_support``: idle or distance-based social interaction strategy.
* ``step_size_support``: fixed or adaptive chemotactic step-size strategy.
* ``population_size``: number of bacteria, at least two. Odd values are
  rounded up to the next even number because reproduction operates in pairs.
* ``chemotactic_steps``: number of sweeps before reproduction.
* ``swim_length``: maximum number of accepted swim moves after a tumble. It
  may be zero.
* ``reproduction_steps``: number of reproduction events before dispersal.
* ``elimination_dispersal_events``: number of completed dispersal events
  before natural termination.
* ``elimination_dispersal_probability``: independent replacement probability
  for each bacterium, in ``[0, 1]``.
* ``finish_control``: evaluation, iteration, and/or time stopping criteria.
* ``problem``: concrete single-objective
  :class:`~uo.problem.problem.Problem`.
* ``solution_template``: concrete
  :class:`~uo.solution.solution.Solution` used to initialize and copy
  bacteria.
* ``output_control``: optional output configuration.
* ``random_seed``: optional seed for BFO's private random generator.
* ``additional_statistics_control``: optional statistics configuration.

.. rubric:: Fixed and adaptive step sizes

``BfoStepSizeSupportFixed(step_size)`` uses the same positive step size for
all movements. ``BfoStepSizeSupportAdaptive`` takes an initial step size,
positive minimum and maximum bounds, and increase and decrease factors. After
an accepted move it applies ``increase_factor``. After a rejected move it
applies ``decrease_factor``. The result is clamped to the configured bounds.

The adaptive value is stored separately for each bacterium. It changes the
chemotactic step size, not the configured ``swim_length``.

.. rubric:: Idle and real swarming

``BfoSwarmingSupportIdle`` returns zero for every interaction. This is the
simplest choice when only the problem fitness should guide the search.

``BfoSwarmingSupportReal`` calls the solution's
``representation_distance`` method and uses four parameters:
``attractant_depth``, ``attractant_width``, ``repellent_height``, and
``repellent_width``. Its result is a temporary adjustment to the effective
fitness and is not stored in the solution.

Example
---------------------

The following example uses bounded real-vector movement. ``problem`` and
``solution_template`` must be concrete application classes whose solution
representation is a list or tuple of real values:

.. code-block:: python

   from uo.algorithm.metaheuristic.bacterial_foraging import (
       BfoOptimizer,
       BfoMovementSupportReal,
       BfoStepSizeSupportAdaptive,
       BfoSwarmingSupportIdle,
   )
   from uo.algorithm.metaheuristic.finish_control import FinishControl

   optimizer = BfoOptimizer(
       movement_support=BfoMovementSupportReal(
           lower_bounds=(-5.0, -5.0),
           upper_bounds=(5.0, 5.0),
       ),
       swarming_support=BfoSwarmingSupportIdle(),
       step_size_support=BfoStepSizeSupportAdaptive(
           initial_step_size=0.25,
           minimum_step_size=0.05,
           maximum_step_size=1.0,
       ),
       population_size=20,
       chemotactic_steps=4,
       swim_length=2,
       reproduction_steps=2,
       elimination_dispersal_events=10,
       elimination_dispersal_probability=0.1,
       finish_control=FinishControl(
           criteria="evaluations",
           evaluations_max=10_000,
       ),
       problem=problem,
       solution_template=solution_template,
       random_seed=2026,
   )
   best_solution = optimizer.optimize()

Stopping, cost, and limitations
---------------------

BFO stops when the configured number of elimination-dispersal events has been
completed or when the inherited
:class:`~uo.algorithm.metaheuristic.finish_control.FinishControl` stops it.
Evaluation limits are checked before candidate evaluations, including swim
moves and dispersed replacements. This means that the last sweep may be
partial when the evaluation limit is reached.

Initializing the population costs ``population_size`` evaluations. A
chemotactic sweep costs between ``population_size`` and
``population_size * (swim_length + 1)`` candidate evaluations, depending on
how many swim moves are accepted. Each dispersed replacement adds one
evaluation.

The built-in movement strategy supports only bounded list and tuple real
vectors. Binary, permutation, and multi-objective representations need another
movement or optimizer strategy. Swim length is fixed, reproduction duplicates
the healthier half, dispersal probability is not quality-dependent, and the
implementation does not include a hybrid local-search step.

API reference
---------------------

See :doc:`uo.algorithm.metaheuristic.bacterial_foraging` for the complete
autodoc reference and exported classes.
