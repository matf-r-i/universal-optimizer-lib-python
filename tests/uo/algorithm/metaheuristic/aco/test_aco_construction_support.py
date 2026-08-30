import unittest

from uo.algorithm.metaheuristic.aco.aco_construction_support import AcoConstructionSupport


class TestAcoConstructionSupport(unittest.TestCase):

    def test_cannot_instantiate_abstract_class_directly(self):
        with self.assertRaises(TypeError):
            AcoConstructionSupport()

    def test_concrete_subclass_can_be_instantiated(self):
        class ConcreteSupport(AcoConstructionSupport):
            def copy(self):
                return ConcreteSupport()

            def construct(self, problem, solution, optimizer):
                pass

            def local_search(self, problem, solution, optimizer):
                pass

        support = ConcreteSupport()

        self.assertIsInstance(support, AcoConstructionSupport)

    def test_subclass_missing_a_method_cannot_be_instantiated(self):
        class IncompleteSupport(AcoConstructionSupport):
            def copy(self):
                return IncompleteSupport()

            def construct(self, problem, solution, optimizer):
                pass

        with self.assertRaises(TypeError):
            IncompleteSupport()


if __name__ == "__main__":
    unittest.main()
