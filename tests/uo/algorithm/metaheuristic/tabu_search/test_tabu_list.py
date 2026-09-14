import unittest

from uo.algorithm.metaheuristic.tabu_search.tabu_list import TabuList


class TestTabuList(unittest.TestCase):

    def test_initialized_with_valid_tenure(self):
        tabu_list = TabuList(tenure=3)
        self.assertEqual(tabu_list.tenure, 3)
        self.assertEqual(len(tabu_list), 0)

    def test_tenure_type_error(self):
        with self.assertRaises(TypeError):
            TabuList(tenure="3")

    def test_tenure_must_be_positive(self):
        with self.assertRaises(ValueError):
            TabuList(tenure=0)
        with self.assertRaises(ValueError):
            TabuList(tenure=-1)

    def test_add_and_contains(self):
        tabu_list = TabuList(tenure=3)
        self.assertFalse(tabu_list.contains((0, 1)))
        tabu_list.add((0, 1))
        self.assertTrue(tabu_list.contains((0, 1)))
        self.assertEqual(len(tabu_list), 1)

    def test_oldest_move_expires_when_tenure_exceeded(self):
        tabu_list = TabuList(tenure=2)
        tabu_list.add((0, 1))
        tabu_list.add((1, 2))
        self.assertTrue(tabu_list.contains((0, 1)))
        tabu_list.add((2, 3))
        self.assertFalse(tabu_list.contains((0, 1)))
        self.assertTrue(tabu_list.contains((1, 2)))
        self.assertTrue(tabu_list.contains((2, 3)))
        self.assertEqual(len(tabu_list), 2)

    def test_copy_returns_independent_copy(self):
        tabu_list = TabuList(tenure=3)
        tabu_list.add((0, 1))

        copied = tabu_list.copy()
        self.assertIsNot(tabu_list, copied)
        self.assertTrue(copied.contains((0, 1)))

        copied.add((1, 2))
        self.assertFalse(tabu_list.contains((1, 2)))
        self.assertTrue(copied.contains((1, 2)))

    def test_string_rep_contains_tenure_and_moves(self):
        tabu_list = TabuList(tenure=2)
        tabu_list.add((0, 1))
        s = str(tabu_list)
        self.assertIn("tenure=2", s)
        self.assertIn("(0, 1)", s)

    def test_repr_and_format(self):
        tabu_list = TabuList(tenure=2)
        self.assertIsInstance(repr(tabu_list), str)
        self.assertIsInstance("{:}".format(tabu_list), str)


if __name__ == "__main__":
    unittest.main()
