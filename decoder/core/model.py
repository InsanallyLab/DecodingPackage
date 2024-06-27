from numpy.typing import ArrayLike
from typing import Union

class Likelihood:
    """ Stores relevant probability information for ISI analysis. """
    def __init__(self):
        """
        Parameters
        ----------
        pdf : scipy interp1d object, or any Python object representing a PDF function.
            Stores the probability density for observing a given ISI.
        inv_cdf : scipy interp1d object, or any Python object representing an inverse CDF function.
            Stores an inverse CDF mapping a probability to ISI range.
        prior_0 : float
            Stores priors such as probabilities of trial conditions. 
        prior_empty : float
            Stores probability of no ISIs in trial.
        """
        self.pdf = None
        self.inv_cdf = None
        self.prior_0 = None
        self.prior_empty = None

    def set_val(
        self, 
        pdf = None, 
        inv_cdf = None, 
        prior_0: Union[int, float] = None, 
        prior_empty: Union[int, float] = None):

        if pdf is not None:
            self.pdf = pdf
        if inv_cdf is not None:
            self.inv_cdf = inv_cdf
        if prior_0 is not None:
            if not isinstance(prior_0, (int, float)):
                raise TypeError("Unsupported data type for prior_0")
            self.prior_0 = prior_0
        if prior_empty is not None:
            if not isinstance(prior_empty, (int, float)):
                raise TypeError("Unsupported data type for prior_empty")
            self.prior_empty = prior_empty

    def __str__(self) -> str:
        return f"PDF: {self.pdf}, Inv_CDF: {self.inv_cdf}, Prior_0: {self.prior_0}, Prior_Empty: {self.prior_empty}"
    

class Model:
    """
    Stores all relevant unconditional and conditional probability information
    for observing given ISIs. 

    Attributes
    ----------
    self.all : a Likelihood object for unconditional ISI probabiities
        self.all.pdf : represents the probability density for observing a given
        ISI, i.e. p(ISI)
        self.all.cdf : represents the inverse CDF mapping probabilities to a 
        ISI range
        
    self.conds : a dict mapping trial conditions to their corresponding 
    Likelihood objects for conditional ISI probabilities.
        self.conds[condition].pdf : represents the conditional probability 
        density for observing a given ISI on each trial condition, 
        i.e. p(ISI | condition)
        self.conds[condition].cdf : represents the inverse CDF mapping 
        conditional probabilities to a ISI range.
        self.conds[condition].prior_0 : stores the probability of the given 
        trial condition, i.e. p(condition)
        self.conds[condition].prior_empty: stores the probability of no ISIs in
        trial conditioned on the trial condition, i.e. p(no ISIs | condition). 

    """
    def __init__(self, conditions: ArrayLike):
        """
        Parameters
        ----------
        conditions : array-like 
            Stores all the possible trial conditions as strings.
            For example, ["target", "non-target"] for stimulus category, or
            ["go", "no-go"] for behavioral choice.
        """
        self.all = Likelihood()
        self.conds = {}
        for cond in conditions:
            if not isinstance(cond, str):
                raise TypeError("Trial conditions must be strings")
            self.conds[cond] = Likelihood()

    def set_all(self, pdf=None, inv_cdf=None): 
        self.all.set_val(pdf, inv_cdf) 

    def set_cond(
        self, 
        cond: str, 
        pdf=None, 
        inv_cdf=None, 
        prior_0: Union[int, float] = None, 
        prior_empty: Union[int, float] = None): 
        if cond not in self.conds:
            raise ValueError("Condition %s not in model's set of conditions" %cond)
        self.conds[cond].set_val(pdf, inv_cdf, prior_0, prior_empty)

    def __str__(self) -> str:
        conds_str = "\n".join(f"{label}: {likelihood}" for label, likelihood in self.conds.items())
        return f"Model:\nAll: {self.all}\nConditions:\n{conds_str}"