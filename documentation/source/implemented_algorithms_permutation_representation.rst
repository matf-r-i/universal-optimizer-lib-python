.. _Algorithm_Permutation_Representation:

Permutation representation
===========================

Basic information
------------------

A permutation representation encodes a solution as an ordering. The native representation is a
``list[int]``, and the library supports both a permutation in the usual sense, in which every
value occurs once, and a **permutation with repetition**, in which a value occurs as many times
as the problem requires. The latter is the encoding of Bierwirth [Bierwirth1995]_ for scheduling
problems: the `c`-th occurrence of value `j` denotes the `c`-th operation of job `j`, so the order
of operations within a job is respected by construction.

What makes the encoding suitable for metaheuristics is that **every** list holding the required
multiset of values encodes a feasible solution. No repair is ever needed, and the elementary move
of the representation, a swap of two positions, always produces another feasible solution.

Before this support was added, the library offered neighborhoods and genetic operators for the
``BitArray`` and the ``int`` representations only. Those are inapplicable to an ordering: flipping
a bit or inverting a position of an integer destroys the multiset of values, so the result is not
a permutation any more.

Implementation notes
---------------------

All supports skip a swap of two positions that hold **equal** values. For a permutation in the
usual sense no such pair exists, and for a permutation with repetition such a swap would leave the
solution unchanged while still being charged as an evaluation of the objective function.

Simulated annealing
^^^^^^^^^^^^^^^^^^^^

:class:`~uo.algorithm.metaheuristic.simulated_annealing.sa_neighborhood_permutation.SaNeighborhoodPermutation`
generates a neighbor by applying ``k`` successive swaps; ``k`` defaults to one. The neighbor is a
new solution, and the original is left intact. See :ref:`Algorithm_Simulated_Annealing`.

Variable neighborhood search
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~uo.algorithm.metaheuristic.variable_neighborhood_search.vns_shaking_support_standard_permutation.VnsShakingSupportStandardPermutation`
shakes the solution in place by applying ``k`` successive swaps, where ``k`` is the index of the
neighborhood supplied by the optimizer.

Local search comes in two variants,
:class:`~uo.algorithm.metaheuristic.variable_neighborhood_search.vns_ls_support_standard_fi_permutation.VnsLocalSearchSupportStandardFirstImprovementPermutation`
and
:class:`~uo.algorithm.metaheuristic.variable_neighborhood_search.vns_ls_support_standard_bi_permutation.VnsLocalSearchSupportStandardBestImprovementPermutation`.
Both explore the neighborhood formed by **all** swaps of two positions, which does not depend on
``k``. For the binary representations the `k`-th neighborhood inverts `k` bits; the corresponding
construction over permutations would enumerate `k` successive swaps, which grows as the `2k`-th
power of the dimension and is not usable even for moderate instances. Parameter ``k`` is therefore
used only for the range check, while diversification is left to the shaking support.
See :ref:`Algorithm_Variable_Neighborhood_Search`.

Genetic algorithm
^^^^^^^^^^^^^^^^^^

:class:`~uo.algorithm.metaheuristic.genetic_algorithm.ga_mutation_support_swap_permutation.GaMutationSupportSwapPermutation`
visits every position of the permutation and, with the supplied probability, swaps it with another
position drawn uniformly at random.

:class:`~uo.algorithm.metaheuristic.genetic_algorithm.ga_crossover_support_ppx_permutation.GaCrossoverSupportPpxPermutation`
implements the precedence preservative crossover (PPX) of Bierwirth, Mattfeld and Kopfer
[Bierwirth1996]_. A one point or a uniform crossover applied directly to two permutations would in
general produce a child in which some values appear too often and others not at all. PPX avoids
that by never writing a value into the child directly: both parents are consumed from the front,
a random draw decides from which parent the leading value is taken, that value is appended to the
child and then removed from **both** parents. Every value is therefore consumed exactly as many
times as it occurs, so the multiset of the child equals the multiset of either parent, while the
relative order of the values inherited from a parent is preserved. Crossing an individual with
itself reproduces that individual, which is the invariant the implementation is tested against.

:class:`~uo.algorithm.metaheuristic.genetic_algorithm.ga_selection_tournament.GaSelectionTournament`
selects the winner of a tournament among ``tournament_size`` individuals drawn without
replacement. Unlike fitness proportional selection it uses no global statistic of the population,
only comparisons within the drawn group, so it is indifferent to the scale and the sign of the
fitness. That matters for minimization problems, whose fitness is the negated objective value.
See :ref:`Algorithm_Genetic_Algorithm`.

Example
-------

.. code-block:: python

   optimizer = SaOptimizer(
       sa_neighborhood=SaNeighborhoodPermutation(problem.dimension, k=1),
       sa_temperature=SaTemperatureExponential(0.9, 0.9995),
       finish_control=FinishControl(criteria="evaluations", evaluations_max=20000),
       problem=problem,
       solution_template=solution,
       random_seed=43434343,
   )
   best_solution = optimizer.optimize()

Parameters and limitations
---------------------------

Every support is created for a fixed ``dimension``, the length of the permutation, and rejects a
solution whose representation is of a different length. That catches a support built for one
instance and then applied to another, which would otherwise pass silently and produce meaningless
results.

A swap is the only move that is implemented. Moves that are known to work well on some ordering
problems, such as the insertion move or the inversion of a whole segment, are not provided.
``GaCrossoverSupportPpxPermutation`` requires both parents to hold the same multiset of values and
raises otherwise.

API reference
-------------

See :doc:`uo.algorithm.metaheuristic.simulated_annealing`,
:doc:`uo.algorithm.metaheuristic.variable_neighborhood_search` and
:doc:`uo.algorithm.metaheuristic.genetic_algorithm`.

Problems solved with this representation
-----------------------------------------

The Job Shop Scheduling Problem of the application uses this representation; see the page
`Job Shop Scheduling Problem
<https://matf-r-i.github.io/universal-optimizer-app/problems_to_be_solved_job_shop_scheduling_problem.html>`_.

References
----------

.. [Bierwirth1995] Bierwirth, C. (1995). "A generalized permutation approach to job shop scheduling with genetic algorithms". OR Spektrum. 17 (2-3): 87-92.

.. [Bierwirth1996] Bierwirth, C.; Mattfeld, D. C.; Kopfer, H. (1996). "On permutation representations for scheduling problems". Parallel Problem Solving from Nature IV. Lecture Notes in Computer Science 1141: 310-318.

