.. _Algorithm_Ant_Colony_Optimization:

Ant Colony Optimization
========================

Basic information
------------------

Ant Colony Optimization (ACO) is a population-based metaheuristic inspired
by ant foraging behavior. Ants construct candidate solutions step by step,
guided by pheromone trails and a problem-specific heuristic. Trails on
better solutions are reinforced; all trails evaporate over time, keeping
the colony exploring.

This implementation is the MMAS (Max-Min Ant System) variant, which bounds
pheromone values within ``[tau_min, tau_max]`` to prevent premature
convergence and starvation.

Implementation notes
---------------------

The implementation is provided by
:class:`~uo.algorithm.metaheuristic.aco.aco_optimizer.AcoOptimizer`, which
holds the shared pheromone/heuristic state, and its concrete subclass
:class:`~uo.algorithm.metaheuristic.aco.aco_optimizer_standard.AcoOptimizerStandard`,
which implements the main loop.

Solution construction and local search are problem-specific and are provided
by implementations of
:class:`~uo.algorithm.metaheuristic.aco.aco_construction_support.AcoConstructionSupport`.
Pheromone evaporation, deposit and clamping are provided by implementations
of :class:`~uo.algorithm.metaheuristic.aco.aco_evaporation_support.AcoEvaporationSupport`;
two strategies are available:
:class:`~uo.algorithm.metaheuristic.aco.aco_evaporation_support.AcoEvaporationSupportFixed`
(deterministic control, deposit source switches at a fixed iteration count)
and
:class:`~uo.algorithm.metaheuristic.aco.aco_evaporation_support.AcoEvaporationSupportAdaptive`
(adaptive control, evaporation rate and deposit source react to the search's
stagnation counter).

Example
-------

.. code-block:: python

   optimizer = AcoOptimizerStandard(
       construction_support=construction_support,
       evaporation_support=AcoEvaporationSupportAdaptive(rho_scale=2.0),
       n_ants=problem.n,
       alpha=1.0,
       beta=2.0,
       rho=0.02,
       p_best=0.05,
       stagnation_limit=100,
       finish_control=FinishControl(criteria="iterations", iterations_max=200),
       problem=problem,
       solution_template=solution,
       output_control=None,
       random_seed=43434343,
       additional_statistics_control=None,
   )
   best_solution = optimizer.optimize()

Parameters and limitations
---------------------------

``alpha`` and ``beta`` control the relative influence of pheromone versus
heuristic desirability during construction. ``rho`` is the base evaporation
rate. ``p_best`` derives the MMAS pheromone lower bound; it requires
``p_best ** (1 / n) >= 2 / n`` to keep ``tau_min <= tau_max``, which holds
for every problem instance of realistic size but can fail for very small
``n``. ``stagnation_limit`` controls both the adaptive evaporation strategy
and the hard pheromone reset. The current implementation targets
permutation-based construction (a ``representation.tour`` and a
``tour_cost`` method are expected on the solution) and does not support
multi-objective problems.

API reference
-------------

See :doc:`uo.algorithm.metaheuristic.aco`.

References
----------

.. [DorigoStutzle2004] Dorigo, M.; Stützle, T. (2004). "Ant Colony Optimization". MIT Press.

.. [StutzleHoos2000] Stützle, T.; Hoos, H. H. (2000). "MAX-MIN Ant System". Future Generation Computer Systems. 16 (8): 889-914.

.. [Mavrovouniotis2014] Mavrovouniotis, M.; Yang, S. (2014). "Ant Colony Optimization with Self-Adaptive Evaporation Rate in Dynamic Environments". IEEE Symposium Series on Computational Intelligence.
