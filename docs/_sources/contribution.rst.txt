How to Contribute
=================
    

This system is developed in `Python <https://www.python.org>`_ programming language, using `poetry <https://python-poetry.org>`_ (alternatively `pip <https://pypi.org/project/pip/>`_) as project and package manager , with `unittest <https://docs.python.org/3/library/unitest.html>`_  library for unit testing and `Sphinx <https://www.sphinx-doc.org/en/master>`_ system for documentation generation. Same tool set should be use for contribution to the project.

Contribution is encouraged in designing novel optimization methods. Requirements:

    1. Algorithms should be derived from the specified class.

        - Class that implements metaheuristic optimization should be derived either from the :class:`uo.algorithm.metaheuristic.single_solution_metaheuristic.SingleSolutionMetaheuristic` class, or from the :class:`uo.algorithm.metaheuristic.population_based_metaheuristic.PopulationBasedMetaheuristic`. It should be placed into separate directory within :file:`/uo/algorithm/metaheuristic/` directory.

        - Class that implements exact optimization should be derived from the :class:`uo.algorithm.Algorithm` class. That class should be placed into separate directory within :file:`/uo/algorithm/` directory.

    2. Type hints and documentation.

        - All programming objects (classes, functions, variables, parameters, optional parameters etc.) should be `type-hinted <https://www.infoworld.com/article/3630372/get-started-with-python-type-hints.html>`_
        
        - All programming objects (classes, functions, etc.) should be properly documented using the system `Sphinx`, reStructuredText and doc comments within the code.

        - Each of the implemented algorithm should have separate documentation web page, where that algorithm is described and documented. At least, there should be the link from doc comments within implemented functionality toward the web page that explains algorithm and vice versa.  

    3. Unit testing coverage.
    
        - Implemented programming code should be fully covered with unit tests.  
    
        - Here, `unittest` framework  used. 
        
        - Test should be placed into separate sub-directory under :file:`/uo/tests/` directory. Directory structure within :file:`/uo/tests/` directory should mirror directory structure of the :file:`/uo/` directory.  

        - All developed code should be covered with unit test, and test coverage rate should be not less than 80%. 


Contributors
============

Contribution domains
--------------------

Contribution in the designing novel **optimization methods**:

    a.1. Library and application:
    
        1. Initial overall structure and organization - [VladimirFilipovic]_

    a.2. Total Enumeration (TE) exact algorithm: 
    
        2. Structure, organization and main loop implementation - [VladimirFilipovic]_ 

        3. Implementation with bit-array based complex counters (class :class:`~uo.utils.ComplexCounterBitArrayFull`, using `bitstring.BitArray` class) - [VladimirFilipovic]_

        4. Implementation with int based complex counters (classes :class:`~uo.utils.ComplexCounterUniformFull` and :class:`~uo.utils.ComplexCounterUniformAscending`, using `int` values) - [VladimirFilipovic]_

    a.3. Variable Neighborhood Search :ref:`Algorithm_Variable_Neighborhood_Search` (VNS) metaheuristics:
        
        5. Structure, organization and main loop implementation - [VladimirFilipovic]_ 

        6. Implementation of shaking and local searches with binary representation  (in class :class:`~uo.algorithm.variable_neighborhood_search.VnsShakingSupportStandardInt`, using `int` predefined type) - [VladimirFilipovic]_ 

        7. Implementation of shaking and local searches with binary representation (in class :class:`~uo.algorithm.variable_neighborhood_search.VnsShakingSupportStandardBitArray`,using :class:`bitstring.BitArray` class) - [VladimirFilipovic]_ 

    a.4. Genetic Algorithms :ref:`Algorithm_Genetic_Algorithm` (GA) metaheuristics:
        
        8. Structure, organization and main loop implementation - [MarkoRadosavljevic]_, [VladimirFilipovic]_ 

        9. Making class :class:`uo.algorithm.metaheuristic.genetic_algorithm.GaOptimizer` to be abstract and dividing its functionality into non-abstract classes :class:`uo.algorithm.metaheuristic.genetic_algorithm.GaOptimizerGenerational` and :class:`uo.algorithm.metaheuristic.genetic_algorithm.GaOptimizerSteadyState` - [VladimirFilipovic]_ 

        10. Implementation of GA selection methods (in classes: :class:`~uo.algorithm.metaheuristic.genetic_algorithm.GaSelectionIdle`, :class:`~uo.algorithm.metaheuristic.genetic_algorithm.GaSelectionRoulette`)  - [MarkoRadosavljevic]_

        11. Implementation of GA crossover one point method (contained within class: :class:`~uo.algorithm.metaheuristic.genetic_algorithm.GaCrossoverSupportOnePointBitArray`), with binary representation (using `bitstring.BitArray` class) - [MarkoRadosavljevic]_ 

        12. Implementation of GA mutation one point method (contained within class: :class:`~uo.algorithm.metaheuristic.genetic_algorithm.GaMutationSupportOnePointBitArray`), with binary representation (using `bitstring.BitArray` class) - [MarkoRadosavljevic]_ 

    a.4. Electromagnetism-like :ref:`Algorithm_Electromagnetism_Like_Metaheuristic` (EM) metaheuristics:
        
        13. Structure, organization and main loop implementation - [AndjelaDamnjanovic]_

    a.5. Simulated Annealing :ref:`Algorithm_Simulated_Annealing` (SA) metaheuristics:
        
        14. Structure, organization and main loop implementation - [MarkoLazarevic]_ 

        15. Implementation of neighborhood structures with integer representation (in class :class:`~uo.algorithm.metaheuristic.simulated_annealing.SaNeighborhoodInt`, using `int` predefined type) - [MarkoLazarevic]_ 

        16. Implementation of neighborhood structures with binary representation (in class :class:`~uo.algorithm.metaheuristic.simulated_annealing.SaNeighborhoodBitArray`, using :class:`bitstring.BitArray` class) - [MarkoLazarevic]_

        17. Implementation of temperature calculating methods (in classes: :class:`~uo.algorithm.metaheuristic.simulated_annealing.SaTemperatureConst`, :class:`~uo.algorithm.metaheuristic.simulated_annealing.SaTemperatureLinear`, :class:`~uo.algorithm.metaheuristic.simulated_annealing.SaTemperatureExponential`) - [MarkoLazarevic]_ 

    a.6. Model-Agnostic Meta-Learning :doc:`MAML <maml_metaheuristic>` (MAML) metaheuristics:
        
        18. Structure, organization and main loop implementation - [StojanKostic]_

        19. Implementation of numerical gradient calculation using finite-difference method - [StojanKostic]_


Contributor List
----------------

.. [VladimirFilipovic] Vladimir Filipović, `<https://github.com/vladofilipovic>`_ e-mail: vladofilipovic@hotmail.com

.. [MarkoRadosavljevic] Marko Radosavljević, `<https://github.com/Markic01>`_ e-mail: mi20079@alas.matf.bg.ac.rs

.. [AndjelaDamnjanovic] Anđela Damjanović, `<https://github.com/AndjelaDamnjanovic>`_ e-mail: mi19059@alas.matf.bg.ac.rs

.. [LazarSavic] Lazar Savić, `<https://github.com/killica>`_ e-mail: mi21004@alas.matf.bg.ac.rs

.. [MarkoLazarevic] Marko Lazarević, `<https://github.com/marko-lazarevic>`_ e-mail: mi21098@alas.matf.bg.ac.rs

.. [StojanKostic] Stojan Kostić, `<https://github.com/Stojan-Kole>`_ e-mail: mi21131@alas.matf.bg.ac.rs
