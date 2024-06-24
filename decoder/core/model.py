import joblib

class Likelihood:
    """
    Likelihood initializer.

    Parameters
    ----------
    pdf : scipy interp1d object, or any Python object representing a PDF function.
    inv_cdf : scipy interp1d object, or any Python object representing a PDF function.
    prior_0 : float
    prior_empty : float
    """
    def __init__(self) -> None:
        self.pdf = None
        self.inv_cdf = None
        self.prior_0 = None
        self.prior_empty = None

    def set_val(self, pdf=None, inv_cdf=None, prior_0=None, prior_empty=None):
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

    def __str__(self):
        return f"PDF: {self.pdf}, Inv_CDF: {self.inv_cdf}, Prior_0: {self.prior_0}, Prior_Empty: {self.prior_empty}"
    

class Model:
    def __init__(self, conditions) -> None:
        self.all = Likelihood()
        self.conds = {}
        for label in conditions:
            self.conds[label] = Likelihood()

    def set_all(self, pdf=None, inv_cdf=None): 
        self.all.set_val(pdf, inv_cdf) 

    def set_cond(self, cond, pdf=None, inv_cdf=None, prior_0=None, prior_empty=None): 
        if cond not in self.conds:
            raise ValueError("Condition %s not in model's set of conditions" %cond)
        self.conds[cond].set_val(pdf, inv_cdf, prior_0, prior_empty)
 
    def save_model(self, filename):
        joblib.dump(self, filename)

    def __str__(self):
        conds_str = "\n".join(f"{label}: {likelihood}" for label, likelihood in self.conds.items())
        return f"Model:\nAll: {self.all}\nConditions:\n{conds_str}"