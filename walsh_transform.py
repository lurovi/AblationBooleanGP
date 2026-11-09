import numpy as np
from full_binary_domain import FullBinaryDomain


class WalshTransform:
    def __init__(self,
                 n_bits: int
                 ) -> None:
        self.__domain: FullBinaryDomain = FullBinaryDomain(n_bits)
        self.__number_of_ones_for_each_number: np.ndarray = np.array([bin(i)[2:].count('1') for i in range(self.__domain.space_cardinality())])
        self.__boolean_mask_number_of_ones_for_each_number: dict[int, np.ndarray] = {t: self.__number_of_ones_for_each_number <= t for t in range(self.__domain.number_of_bits() + 1)}

    def domain(self) -> FullBinaryDomain:
        return self.__domain

    def number_of_ones_for_each_number(self) -> np.ndarray:
        return self.__number_of_ones_for_each_number

    def boolean_mask_number_of_ones_for_each_number(self, t: int) -> np.ndarray:
        return self.__boolean_mask_number_of_ones_for_each_number[t]

    def resiliency(self, spectrum: np.ndarray) -> int:
        max_resiliency_found_so_far: int = -1
        t: int = 0
        if spectrum[0] == 0:
            max_resiliency_found_so_far = 0
            for iii in range(self.__domain.number_of_bits()):
                t += 1
                m: np.ndarray = self.boolean_mask_number_of_ones_for_each_number(t)
                if np.any(spectrum[m]):
                    return max_resiliency_found_so_far
                else:
                    max_resiliency_found_so_far += 1
                    if iii == self.__domain.number_of_bits() - 1:
                        return max_resiliency_found_so_far
        else:
            return max_resiliency_found_so_far
        return -1

    def correlation_immunity(self, spectrum: np.ndarray, tol: float = 1e-9) -> int:
        size = 1 << self.__domain.number_of_bits()
        assert spectrum.shape[0] == size, "Walsh spectrum length mismatch"

        max_order = 0
        # For each possible order
        for m in range(1, self.__domain.number_of_bits() + 1):
            ok = True
            for idx in range(1, size):  # exclude index 0
                if bin(idx).count("1") <= m and abs(spectrum[idx]) > tol:
                    ok = False
                    break
            if ok:
                max_order = m
            else:
                break
        return max_order

    def non_linearity(self, spectrum: np.ndarray) -> float:
        x: np.ndarray = np.absolute(spectrum)
        m: float = float(np.max(x))
        return  self.__domain.space_cardinality() / 2.0 - 0.5 * m

    def granular_non_linearity(self, spectrum: np.ndarray) -> float:
        x: np.ndarray = np.absolute(spectrum)
        m: float = float(np.max(x))
        max_values: int = int((x == m).sum())
        nl: float = self.__domain.space_cardinality() / 2.0 - 0.5 * m
        closeness: float = (self.__domain.space_cardinality() -  max_values) / self.__domain.space_cardinality()
        return nl + closeness
    
    def granular_non_linearity_with_balancedness_penalty(self, truth_table: np.ndarray, spectrum: np.ndarray) -> float:
        nl_granular: float = self.granular_non_linearity(spectrum)
        balancedness_penalty: int = self.__domain.balancedness_penalty(truth_table)
        return - balancedness_penalty + int(balancedness_penalty == 0) * nl_granular

    def apply(self, result: np.ndarray) -> tuple[np.ndarray, int]:
        return self.__fast_walsh_transform_vectorized(result)

    def invert(self, spectrum: np.ndarray, directly_go_to_truth_table: bool = False) -> tuple[np.ndarray, int]:
        return self.__inverse_fast_walsh_transform_vectorized(spectrum, directly_go_to_truth_table)
    
    def __fast_walsh_transform_vectorized(self, result: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Fully vectorized Fast Walsh Transform using numpy operations.
        No Python loops - processes all butterflies in parallel at each stage.
        
        Args:
            result: Truth table to transform
            
        Returns:
            Tuple of (spectrum, spectral_radius)
        """
        # Convert to polar form
        polar_form: np.ndarray = FullBinaryDomain.convert_truth_table_to_polar_form(result)
        v: np.ndarray = polar_form.copy().astype(np.int64)
        n: int = len(v)
        
        # Iterative Walsh-Hadamard transform
        # Process in log2(n) stages, each stage processes all butterflies in parallel
        h: int = 1  # Current butterfly distance
        while h < n:
            # Reshape to process butterflies in parallel
            # Split array into blocks of size 2*h, then separate first/second halves
            v_reshaped = v.reshape(n // (2 * h), 2 * h)
            
            # Split each block into two halves
            first_half = v_reshaped[:, :h].copy()
            second_half = v_reshaped[:, h:2*h]
            
            # Apply butterfly operation in parallel (vectorized)
            # (a, b) -> (a + b, a - b)
            v_reshaped[:, :h] = first_half + second_half
            v_reshaped[:, h:2*h] = first_half - second_half
            
            h *= 2
        
        # Compute spectral radius (max absolute value)
        spectral_radius: int = int(np.max(np.abs(v)))
        
        return v, spectral_radius
    
    def __inverse_fast_walsh_transform_vectorized(self, result: np.ndarray, directly_go_to_truth_table: bool = False) -> tuple[np.ndarray, int]:
        """
        Fully vectorized Inverse Fast Walsh Transform using numpy operations.
        No Python loops - processes all butterflies in parallel at each stage.
        
        Args:
            result: Spectrum to inverse transform
            directly_go_to_truth_table: Whether to convert result to truth table
            
        Returns:
            Tuple of (transformed_array, max_autocorrelation_coefficient)
        """
        v: np.ndarray = result.copy().astype(np.int64)
        n: int = len(v)
        
        # Iterative inverse Walsh-Hadamard transform
        h: int = 1
        while h < n:
            # Reshape to process butterflies in parallel
            v_reshaped = v.reshape(n // (2 * h), 2 * h)
            
            # Split each block into two halves
            first_half = v_reshaped[:, :h].copy()
            second_half = v_reshaped[:, h:2*h]
            
            # Apply inverse butterfly operation in parallel with division by 2
            # (a, b) -> ((a + b) / 2, (a - b) / 2)
            v_reshaped[:, :h] = (first_half + second_half) // 2
            v_reshaped[:, h:2*h] = (first_half - second_half) // 2
            
            h *= 2
        
        # Compute max autocorrelation coefficient (max abs value excluding index 0)
        if n > 1:
            max_auto_correlation_coefficient: int = int(np.max(np.abs(v[1:])))
        else:
            max_auto_correlation_coefficient: int = 0
        
        if not directly_go_to_truth_table:
            return v, max_auto_correlation_coefficient
        return FullBinaryDomain.convert_polar_form_to_truth_table(v), max_auto_correlation_coefficient

    # ========== Old recursive implementations (kept for reference) ==========
    
    def __fast_walsh_transform_init(self, result: np.ndarray) -> tuple[np.ndarray, int]:
        polar_form: np.ndarray = FullBinaryDomain.convert_truth_table_to_polar_form(result)
        l: list[int] = polar_form.tolist()
        spectral_radius: int = self.__fast_walsh_transform(l, 0, len(l))
        return np.array(l), spectral_radius

    def __fast_walsh_transform(self, v: list[int], start: int, length: int) -> int:
        half: int = length // 2
        for i in range(start, start + half):
            temp: int = v[i]
            v[i] += v[i + half]
            v[i + half] = temp - v[i + half]

        if half > 1:
            val1: int = self.__fast_walsh_transform(v, start, half)
            val2: int = self.__fast_walsh_transform(v, start + half, half)
            return max(val1, val2)
        else:
            if abs(v[start]) > abs(v[start + half]):
                return abs(v[start])
            else:
                return abs(v[start + half])

    def __inverse_fast_walsh_transform_init(self, result: np.ndarray, directly_go_to_truth_table: bool = False) -> tuple[np.ndarray, int]:
        l: list[int] = result.tolist()
        max_auto_correlation_coefficient: int = self.__inverse_fast_walsh_transform(l, 0, len(l))
        r: np.ndarray = np.array(l, dtype=np.int64)
        if not directly_go_to_truth_table:
            return r, max_auto_correlation_coefficient
        return FullBinaryDomain.convert_polar_form_to_truth_table(r), max_auto_correlation_coefficient

    def __inverse_fast_walsh_transform(self, v: list[int], start: int, length: int) -> int:
        half: int = length // 2
        for i in range(start, start + half):
            temp: int = v[i]
            v[i] = int( (v[i] + v[i + half]) / 2.0 )
            v[i + half] = int( (temp - v[i + half]) / 2.0 )

        if half > 1:
            val1: int = self.__inverse_fast_walsh_transform(v, start, half)
            val2: int = self.__inverse_fast_walsh_transform(v, start + half, half)
            return max(val1, val2)
        else:
            if start == 0:
                return abs(v[1])
            else:
                if abs(v[start]) > abs(v[start + half]):
                    return abs(v[start])
                else:
                    return abs(v[start + half])

