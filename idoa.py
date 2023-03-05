import numpy as np
from overlap import Overlap
from dissimilarity import Dissimilarity
from scipy.stats import linregress

class IDOA:
    """
    This class calculates the IDOA values vector for the second cohort with respect to the first
    cohort.
    """
    def __init__(self, first_cohort, second_cohort, identical=False, min_overlap=0.5, max_overlap=1, zero_overlap=0.1):
        """
        param first_cohort: the first cohort, samples are in the rows.
        param second_cohort: the second cohort, samples are in the rows.
        param identical: if True, both cohorts are identical.
        param min_overlap: the minimal value of overlap.
        param max_overlap: the maximal value of overlap.
        param zero_overlap: a number, if the maximal value of the overlap vector that calculated
        between sample from the second cohort w.r.t the first cohort is less than min_overlap + zero_overlap
        so the overlap considered to be zero.
        """
        self.first_cohort = first_cohort
        self.second_cohort = second_cohort
        self.num_samples_first = np.size(first_cohort, 0)
        self.num_samples_second = np.size(second_cohort, 0)
        self.IDOA_vector = np.zeros(self.num_samples_second)
        self.identical = identical
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        self.zero_overlap = zero_overlap
        if identical:
            # Identical cohorts not including the dissimilarity and overlap values that calculating with themselves.
            self.overlap_mat = np.zeros((self.num_samples_first, self.num_samples_first-1))
            self.dissimilarity_mat = np.zeros((self.num_samples_first, self.num_samples_first - 1))
        else:
            self.overlap_mat = np.zeros((self.num_samples_first, self.num_samples_first))
            self.dissimilarity_mat = np.zeros((self.num_samples_first, self.num_samples_first))

    def calc_idoa_vector(self):
        """
        This method calculates the vector of the IDOA values that calculated for the second cohort samples w.r.t the
         first cohort for both cases(identical or not).
        :return: IDOA vector.
        """
        if self.identical:
            for i in range(0, self.num_samples_second):
                o_vector = []
                d_vector = []
                for j in range(0, self.num_samples_first):
                    o = Overlap(self.first_cohort[j, :], self.second_cohort[i, :])
                    d = Dissimilarity(self.first_cohort[j, :], self.second_cohort[i, :])
                    o_vector.append(o)
                    d_vector.append(d)
                overlap_vector = np.array([o_vector[j].calculate_overlap()
                                          for j in range(0, self.num_samples_first) if j != i])
                dissimilarity_vector = np.array([d_vector[j].calculate_dissimilarity()
                                                for j in range(0, self.num_samples_first) if j != i])
                if np.max(overlap_vector) <= self.min_overlap + self.zero_overlap:
                    self.IDOA_vector[i] = 0
                else:
                    overlap_vector_index = np.where(np.logical_and(overlap_vector >= overlap_vector >= self.min_overlap,
                                                                   overlap_vector <= self.max_overlap))
                    new_overlap_vector = overlap_vector[overlap_vector_index]
                    new_dissimilarity_vector = dissimilarity_vector[overlap_vector_index]
                    slope = linregress(new_overlap_vector, new_dissimilarity_vector)[0]
                    self.IDOA_vector[i] = slope
            return self.IDOA_vector
        else:
            for i in range(0, self.num_samples_second):
                overlap_vector = np.zeros(self.num_samples_first)
                dissimilarity_vector = np.zeros(self.num_samples_first)
                for j in range(0, self.num_samples_first):
                    o = Overlap(self.first_cohort[j, :], self.second_cohort[i, :])
                    d = Dissimilarity(self.first_cohort[j, :], self.second_cohort[i, :])
                    overlap_vector[j] = o.calculate_overlap()
                    dissimilarity_vector[j] = d.calculate_dissimilarity()
                if np.max(overlap_vector) <= self.min_overlap + self.zero_overlap:
                    self.IDOA_vector[i] = 0
                else:
                    overlap_vector_index = np.where(np.logical_and(overlap_vector >= self.min_overlap,
                                                                   overlap_vector <= self.max_overlap))
                    new_overlap_vector = overlap_vector[overlap_vector_index]
                    new_dissimilarity_vector = dissimilarity_vector[overlap_vector_index]
                    slope = linregress(new_overlap_vector, new_dissimilarity_vector)[0]
                    self.IDOA_vector[i] = slope
            return self.IDOA_vector