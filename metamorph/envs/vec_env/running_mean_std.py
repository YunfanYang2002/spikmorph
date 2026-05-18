import numpy as np


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    if batch_count <= 0:
        return mean, var, count

    mean = np.asarray(mean, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64)
    batch_mean = np.asarray(batch_mean, dtype=np.float64)
    batch_var = np.asarray(batch_var, dtype=np.float64)

    if not np.isfinite(count):
        count = 1e-4
    if not np.all(np.isfinite(mean)):
        mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.all(np.isfinite(var)):
        var = np.nan_to_num(var, nan=1.0, posinf=1.0, neginf=1.0)
    if not np.all(np.isfinite(batch_mean)):
        return mean, var, count
    if not np.all(np.isfinite(batch_var)):
        return mean, var, count

    delta = batch_mean - mean
    tot_count = count + batch_count
    if tot_count <= 0 or not np.isfinite(tot_count):
        return mean, var, count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    new_mean = np.nan_to_num(new_mean, nan=0.0, posinf=0.0, neginf=0.0)
    new_var = np.nan_to_num(new_var, nan=1.0, posinf=1.0, neginf=1.0)

    return new_mean, new_var, new_count


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )
