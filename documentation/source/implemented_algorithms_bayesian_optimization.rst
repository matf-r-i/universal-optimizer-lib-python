.. _Algorithm_Bayesian_Optimization:

Bayesian Optimization
=====================

Basic information
-----------------

Bayesian optimization is a surrogate-based black-box optimization algorithm.
It is intended for expensive, bounded, continuous, single-objective problems,
especially in low-dimensional search spaces. It is not classified as a
metaheuristic in this library.

The implementation fits a Gaussian process with an RBF kernel to all evaluated
solutions. Expected Improvement selects each new candidate, and multi-start
L-BFGS-B maximizes that acquisition function on a normalized unit cube.

Implementation notes
--------------------

The implementation is provided by
:class:`uo.algorithm.bayesian_optimization.optimizer.BayesianOptimizer`, which
derives directly from :class:`uo.algorithm.algorithm.Algorithm`. The supplied
solution template must accept a one-dimensional NumPy vector in ``init_from``
and evaluate it using the supplied problem.

Universal Optimizer compares solutions by maximizing ``fitness_value``. The
Bayesian model minimizes the negative fitness value, so the same implementation
works for both minimization and maximization problems when their solution class
uses the library fitness convention.

Example
-------

.. code-block:: python

   optimizer = BayesianOptimizer(
       problem=problem,
       solution_template=real_vector_solution,
       bounds=[(-5.0, 5.0), (-5.0, 5.0)],
       evaluation_budget=40,
       number_of_initial_points=6,
       random_seed=17,
   )
   best_solution = optimizer.optimize()

Parameters and limitations
--------------------------

``bounds`` must contain one finite lower/upper pair per dimension. The
``evaluation_budget`` is the authoritative stopping condition and includes the
initial random design. The current implementation does not support constraints,
infeasible evaluations, multiple objectives, categorical variables, or batch
evaluation.

API reference
-------------

See :doc:`uo.algorithm.bayesian_optimization`.
