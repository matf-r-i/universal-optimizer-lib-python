..  _Algorithm_MOEAD:

MOEA/D
======

Basic information
-----------------

Multi-Objective Evolutionary Algorithm based on Decomposition (MOEA/D)
is a population-based metaheuristic for solving multi-objective
optimization problems. The method decomposes a multi-objective problem
into a set of scalar subproblems defined by weight vectors and optimizes
them collaboratively by means of neighborhood-based offspring generation
and replacement.

The implementation added to this library supports both real-valued and
binary solution representations. The algorithm uses Tchebyscheff
scalarization, neighborhood structures induced by distances between
weight vectors, and support classes for offspring generation.

Implementation notes
--------------------

Implementation of that optimization method is given within the class
:ref:`MoeadOptimizer<py_moead_optimizer>`.

Supporting functionality is implemented in the following modules and
classes:

- :ref:`MoeadVariationSupport<py_moead_variation_support>`
- :ref:`MoeadVariationSupportReal<py_moead_variation_support_real>`
- :ref:`MoeadVariationSupportBinary<py_moead_variation_support_binary>`
- :ref:`MOEA/D decomposition utilities<py_moead_decomposition>`
- :ref:`MOEA/D weight utilities<py_moead_weights>`
- :ref:`MOEA/D metrics<py_moead_metrics>`

References
----------

.. [ZLi2007] Zhang, Q.; Li, H. (2007). "MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition". IEEE Transactions on Evolutionary Computation. 11 (6): 712–731.

.. [DebAgr1995] Deb, K.; Agrawal, R. B. (1995). "Simulated Binary Crossover for Continuous Search Space". Complex Systems. 9 (2): 115–148.

.. [Deb2001] Deb, K. (2001). "Multi-Objective Optimization Using Evolutionary Algorithms". Wiley.