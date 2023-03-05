import numpy as np

class Overlap:
    """
    This class calculates the overlap value between two given samples
    """
    def __init__(self, sample_first, sample_second):
        """
        :param sample_first: first sample, 1D array.
        :param sample_second: second sample, 1D array.
        """
        self.sample_first = sample_first
        self.sample_second = sample_second
        [self.normalized_sample_1, self.normalized_sample_2] = self.normalize()
        self.s = self.find_intersection()

    def normalize(self):
        """
        This method normalizes the two samples.
        :return: normalized samples.
        """
        normalized_sample_1 = self.sample_first / np.sum(self.sample_first)  # Normalization of the first sample.
        normalized_sample_2 = self.sample_second / np.sum(self.sample_second)  # Normalization of the second sample.
        return normalized_sample_1, normalized_sample_2

    def find_intersection(self):
        """
        This method finds the shared non-zero indexes of the two samples.
        :return: the set s with represent the intersected indexes
        """
        nonzero_index_1 = np.nonzero(self.normalized_sample_1)  # Find the non-zero index of the first sample.
        nonzero_index_2 = np.nonzero(self.normalized_sample_2)  # Find the non-zero index of the second sample.
        s = np.intersect1d(nonzero_index_1, nonzero_index_2)  # Find the intersection.
        return s

    def calculate_overlap(self):
        """
        This method calculates the overlap between the two samples.
        :return: the overlap value.
        """
        # Calculation of the overlap value between the two samples.
        overlap = np.sum(self.normalized_sample_1[self.s] + self.normalized_sample_2[self.s])/2
        return overlap
