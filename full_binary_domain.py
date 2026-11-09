import numpy as np


class FullBinaryDomain:
    def __init__(self, n_bits: int) -> None:
        if n_bits < 1:
            raise AttributeError(f"Number of bits must be at least 1. Specified {n_bits} instead.")
        self.__n_bits: int = n_bits
        self.__space_cardinality: int = 2 ** n_bits
        self.__covering_radius_bound: int = 2 ** (n_bits - 1) - 2 ** (n_bits // 2 - 1)
        res: list[list[int]] = []
        polar_res: list[list[int]] = []
        for i in range(self.__space_cardinality):
            curr_res: list[int] = [int(j) for j in bin(i)[2:]]
            missing_bits: int = n_bits - len(curr_res)
            curr_res = [0] * missing_bits + curr_res
            res.append(curr_res)
            polar_res.append([1 if j == 0 else -1 for j in curr_res])
        self.__data: np.ndarray = np.array(res)
        self.__polar_data: np.ndarray = np.array(polar_res)
        self.__data_pure_int: np.ndarray = np.arange(self.__space_cardinality, dtype=np.int32).reshape(-1, 1)

    def number_of_bits(self) -> int:
        return self.__n_bits

    def space_cardinality(self) -> int:
        return self.__space_cardinality

    def balancedness_penalty(self, output: np.ndarray) -> int:
        # return the number of bits to be changed to make the output balanced
        num_of_zeros: int = (output == 0).sum()
        return int(abs(num_of_zeros - (self.space_cardinality() - num_of_zeros))) // 2

    def covering_radius_bound(self) -> int:
        return self.__covering_radius_bound

    def siegenthaler_bound(self, t: int) -> int:
        if t < 0:
            raise ValueError(f'resiliency order t must be at least 0, found {t} instead.')
        return self.__n_bits - t - 1

    def tarannikov_bound(self, t: int) -> int:
        if t < 0:
            raise ValueError(f'resiliency order t must be at least 0, found {t} instead.')
        return 2 ** (self.__n_bits - 1) - 2 ** (t + 1)

    def data(self) -> np.ndarray:
        return self.__data

    def polar_data(self) -> np.ndarray:
        return self.__polar_data

    def integers(self) -> np.ndarray:
        return self.__data_pure_int

    def balancing(self, output: np.ndarray) -> int:
        num_of_zeros: int = (output == 0).sum()
        return abs(num_of_zeros - (self.space_cardinality() - num_of_zeros))

    def degree(self, output: np.ndarray) -> tuple[list[int], int]:  # Fast Mobius Transform
        truth_table: list[int] = output.tolist()
        length: int = len(truth_table)
        deg: int = self.__degree(truth_table, 0, length)
        return truth_table, deg

    def __degree(self, truth_table: list[int], start: int, length: int) -> int:
        half: int = length // 2
        for i in range(start, start + half):
            truth_table[i + half] = truth_table[i] ^ truth_table[i + half]

        if half > 1:
            val1: int = self.__degree(truth_table, start, half)
            val2: int = self.__degree(truth_table, start + half, half)
            return max(val1, val2)
        else:
            if truth_table[start] == 0 and truth_table[start + half] == 0:
                return 0
            else:
                if truth_table[start + half] == 0:
                    return bin(start)[2:].count('1')
                else:
                    return bin(start + half)[2:].count('1')

    @staticmethod
    def from_numpy_to_binary_string(v: np.ndarray) -> str:
        c: np.ndarray = v.tolist()
        s: str = ""
        for i in c:
            s += str(i)
        return s

    @staticmethod
    def convert_truth_table_to_polar_form(v: np.ndarray) -> np.ndarray:
        return (-1 * v) + (1 - v)

    @staticmethod
    def convert_polar_form_to_truth_table(v: np.ndarray) -> np.ndarray:
        return (1 - v) // 2
