"""
@inproceedings{hundman2018detecting,
  title={Detecting spacecraft anomalies using lstms and nonparametric dynamic thresholding},
  author={Hundman, Kyle and Constantinou, Valentino and Laporte, Christopher and
  Colwell, Ian and Soderstrom, Tom},
  booktitle={Proceedings of the 24th ACM SIGKDD international conference
  on knowledge discovery & data mining},
  pages={387--395},
  year={2018}
}
"""

import numpy as np


class ParametricAnomalyDetector:
    def __init__(self, beta: float = 0.3):
        self.beta = beta

    def detect_anomalies(self, elbo_scores: np.ndarray, p: float = 0.01):
        """
        Returns binary array: 1 = anomaly, 0 = normal.
        """
        errors_s = self.smooth_errors(elbo_scores)
        epsilon = self.find_threshold(errors_s)
        mask = errors_s > epsilon
        sequences = self.get_sequences(mask)
        sequences = self.prune_anomalies(errors_s, sequences, epsilon, p=p)

        labels = np.zeros(len(elbo_scores), dtype=int)
        for start, end in sequences:
            labels[start : end + 1] = 1

        return labels, epsilon, errors_s

    def smooth_errors(self, errors: np.ndarray) -> np.ndarray:
        """Exponentially weighted moving average smoothing."""
        smoothed = np.zeros_like(errors)
        smoothed[0] = errors[0]
        for t in range(1, len(errors)):
            smoothed[t] = self.beta * errors[t] + (1 - self.beta) * smoothed[t - 1]
        return smoothed

    def find_threshold(self, errors_s: np.ndarray, z_range=None) -> float:
        """
        Nonparametric threshold selection.
        Finds z that maximizes relative decrease in mean+std after removing anomalies.
        """
        if z_range is None:
            z_range = np.arange(0.5, 4.0, 0.5)

        mu = errors_s.mean()
        sigma = errors_s.std()
        best_epsilon = mu + z_range[0] * sigma
        best_score = -np.inf

        for z in z_range:
            epsilon = mu + z * sigma
            above = errors_s[errors_s > epsilon]
            below = errors_s[errors_s <= epsilon]

            if len(above) == 0 or len(below) == 0:
                continue

            # Sequences above threshold
            sequences = self.get_sequences(errors_s > epsilon)

            delta_mu = mu - below.mean()
            delta_sigma = sigma - below.std()

            numerator = delta_mu / mu + delta_sigma / sigma
            denominator = len(above) + (len(sequences) ** 2)
            score = numerator / denominator

            if score > best_score:
                best_score = score
                best_epsilon = epsilon

        return best_epsilon

    def get_sequences(self, mask: np.ndarray) -> list:
        """Extract contiguous True sequences from boolean mask."""
        sequences = []
        in_seq = False
        start = 0
        for i, val in enumerate(mask):
            if val and not in_seq:
                start = i
                in_seq = True
            elif not val and in_seq:
                sequences.append((start, i - 1))
                in_seq = False
        if in_seq:
            sequences.append((start, len(mask) - 1))
        return sequences

    def prune_anomalies(
        self, errors_s: np.ndarray, sequences: list, epsilon: float, p: float = 0.01
    ) -> list:
        """
        Remove anomaly sequences where the drop to the next sequence
        is less than p (minimum percent decrease).
        """
        if not sequences:
            return sequences

        # Max error per sequence + first non-anomalous value
        e_max = [errors_s[s : e + 1].max() for s, e in sequences]
        e_max.append(
            errors_s[errors_s <= epsilon].max() if (errors_s <= epsilon).any() else 0
        )
        e_max = sorted(e_max, reverse=True)

        cutoff = len(e_max)
        for i in range(1, len(e_max)):
            if e_max[i - 1] == 0:
                break
            d = (e_max[i - 1] - e_max[i]) / e_max[i - 1]
            if d < p:
                cutoff = i
                break

        threshold_val = e_max[cutoff - 1] if cutoff < len(e_max) else 0
        return [
            (s, e) for s, e in sequences if errors_s[s : e + 1].max() >= threshold_val
        ]
