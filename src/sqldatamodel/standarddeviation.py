### implementation of standard deviation as aggregate function for sqlite ###
from math import sqrt

class StandardDeviation:
    r"""
    Implementation of standard deviation as an aggregate function for SQLite:

    .. math::

        \sigma = \sqrt{\frac{\sum_{i=1}^{N}(x_i - \mu)^2}{N}}

    Where:
        - :math:`x_i` represents each individual data point in the population.
        - :math:`\mu` is the population mean.
        - :math:`N` is the total number of data points in the population.


    This class provides methods to calculate the standard deviation of a set of values
    in an SQLite query using the aggregate function mechanism.

    Attributes:
        ``M`` (float): The running mean of the values.
        ``S`` (float): The running sum of the squared differences from the mean.
        ``k`` (int): The count of non-null input values.

    Note:
        - See :py:mod:`SQLDataModel.describe() <sqldatamodel.sqldatamodel.SQLDataModel.describe>` for statistical implementation.
    """    
    def __init__(self):
        self.M = 0.0
        self.S = 0.0
        self.k = 1

    def step(self, value):
        """
        Update the running mean and sum of squared differences with a new value.

        Parameters:
            ``value`` (float): The input value to be included in the calculation.

        Note:
            - If the input value is None, it will be ignored.
        """        
        if value is None:
            return
        tM = self.M
        self.M += (value - tM) / self.k
        self.S += (value - tM) * (value - self.M)
        self.k += 1

    def finalize(self) -> float:
        """
        Compute the final standard deviation as part of ``sqlite3`` user-defined aggregate function.

        Returns:
            ``float`` or ``None``: The computed standard deviation if the count is greater than or equal to 3, else None.
        
        Note:
            - This returns the population standard deviation, not sample standard deviation. It measures of the spread or dispersion of a set of data points within the population, using the entire population.
        """        
        if self.k < 3:
            return None
        return sqrt(self.S / (self.k-2))