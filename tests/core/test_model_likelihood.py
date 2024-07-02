import pytest
from decoder.core.model import Likelihood, Model
from math import pi, exp 
from scipy.interpolate import interp1d 
import numpy as np

'''
Likelihood unit tests
'''
def normal_pdf(x):
    mu = 0
    sigma = 1
    return 1.0 / (sigma * (2.0 * pi)**(1/2)) * exp(-1.0 * (x - mu)**2 / (2.0 * (sigma**2)))

def uniform_pdf(x):
    a = 0
    b = 1
    return 1.0 / (b - a)

def uniform_inverse_cdf(x):
    a = 0 
    b = 1
    return a + (b - a)*x

def test_likelihood_init():
    likelihood = Likelihood()
    assert likelihood.pdf is None
    assert likelihood.inv_cdf is None
    assert likelihood.prior_0 is None
    assert likelihood.prior_empty is None

def test_likelihood_set_pdf_function():
    likelihood = Likelihood()
    likelihood.set_val(pdf=normal_pdf)

    assert likelihood.pdf(0) == normal_pdf(0)
    assert likelihood.inv_cdf is None
    assert likelihood.prior_0 is None
    assert likelihood.prior_empty is None

def test_likelihood_set_cdf_function():
    likelihood = Likelihood()
    likelihood.set_val(inv_cdf=uniform_inverse_cdf)

    assert likelihood.pdf is None
    assert likelihood.inv_cdf(0.5) == uniform_inverse_cdf(0.5)
    assert likelihood.prior_0 is None
    assert likelihood.prior_empty is None

def test_likelihood_set_pdf_interp1d():
    x = [0, 1, 2, 3]
    y = [0, 1, 2, 3]
    
    pdf_interpolator = interp1d(x, y, kind='linear', assume_sorted=True)

    likelihood = Likelihood()
    likelihood.set_val(pdf=pdf_interpolator)

    inp = [0, 1, 2]
    assert np.array_equal(likelihood.pdf(inp), pdf_interpolator(inp))
    assert likelihood.inv_cdf is None
    assert likelihood.prior_0 is None
    assert likelihood.prior_empty is None

def test_likelihood_set_cdf_interp1d():
    x = [0, 1, 2, 3]
    y = [0, 1, 2, 3]
    cdf_inverse_interpolator = interp1d(y, x, kind='linear', assume_sorted=True)

    likelihood = Likelihood()
    likelihood.set_val(inv_cdf=cdf_inverse_interpolator)

    inp = [1, 2]
    assert likelihood.pdf is None
    assert np.array_equal(likelihood.inv_cdf(inp), cdf_inverse_interpolator(inp))
    assert likelihood.prior_0 is None
    assert likelihood.prior_empty is None

def test_likelihood_set_pdf_multiple():
    likelihood = Likelihood()
    likelihood.set_val(pdf=normal_pdf)
    assert likelihood.pdf(0) == normal_pdf(0)
    assert likelihood.inv_cdf is None
    assert likelihood.prior_0 is None
    assert likelihood.prior_empty is None

    prior_0 = 0.82
    likelihood.set_val(pdf=uniform_pdf, prior_0=prior_0)
    assert likelihood.pdf(0) == uniform_pdf(0)
    assert likelihood.inv_cdf is None
    assert likelihood.prior_0 == prior_0
    assert likelihood.prior_empty is None

def test_likelihood_set_prior0():
    likelihood = Likelihood()
    prior_0 = 0.34
    likelihood.set_val(inv_cdf=uniform_inverse_cdf, prior_0=prior_0)

    assert likelihood.pdf is None
    assert likelihood.inv_cdf(0.2) == uniform_inverse_cdf(0.2)
    assert likelihood.prior_0 == 0.34
    assert likelihood.prior_empty is None

def test_likelihood_set_prior_empty():
    likelihood = Likelihood()
    prior_empty = 1
    likelihood.set_val(pdf=normal_pdf, inv_cdf=uniform_inverse_cdf, prior_empty=prior_empty)

    assert likelihood.pdf(0) == normal_pdf(0)
    assert likelihood.inv_cdf(0.75) == uniform_inverse_cdf(0.75)
    assert likelihood.prior_0 is None
    assert likelihood.prior_empty == prior_empty

def test_likelihood_setting_prior0_unsupported():
    likelihood = Likelihood()
    prior_0 = "0.4"
    with pytest.raises(TypeError) as exc_info:
        likelihood.set_val(prior_0=prior_0)
    assert str(exc_info.value) == "Unsupported data type for prior_0"

def test_likelihood_setting_prior_empty_unsupported():
    likelihood = Likelihood()
    prior_empty = [0.3]
    with pytest.raises(TypeError) as exc_info:
        likelihood.set_val(prior_empty=prior_empty)
    assert str(exc_info.value) == "Unsupported data type for prior_empty"


'''
Model unit tests
'''
def test_model_init():
    conditions = ['cond1', 'cond2']
    model = Model(conditions=conditions)
    assert len(model.conds) == 2
    assert all(condition in model.conds for condition in conditions)
    assert isinstance(model.all, Likelihood)

def test_model_init_no_conds():
    with pytest.raises(TypeError):
        model = Model()

def test_model_init_empty_conds():
    conditions = []
    model = Model(conditions=conditions)
    assert len(model.conds) == 0
    assert isinstance(model.all, Likelihood)

def test_model_non_unique_conds():
    conditions = ['cond1', 'cond1']
    model = Model(conditions=conditions)
    assert len(model.conds) == 1
    assert all(condition in model.conds for condition in conditions)
    assert isinstance(model.all, Likelihood)

def test_model_set_all():
    model = Model(['cond1', 'cond2'])
    model.set_all(pdf=uniform_pdf, inv_cdf=uniform_inverse_cdf)
    assert model.all.pdf(0) == uniform_pdf(0)
    assert model.all.inv_cdf(0) == uniform_inverse_cdf(0)

def test_model_set_all_pdf():
    model = Model(['cond1', 'cond2'])
    model.set_all(pdf=uniform_pdf)
    assert model.all.pdf(2) == uniform_pdf(2)
    assert model.all.inv_cdf is None

def test_model_set_all_cdf():
    model = Model(['cond1', 'cond2'])
    model.set_all(inv_cdf=uniform_inverse_cdf)
    assert model.all.pdf is None
    assert model.all.inv_cdf(0.4) == uniform_inverse_cdf(0.4)

def test_model_set_cond():
    model = Model(['cond1', 'cond2'])
    prior_empty = 0.1
    cond = "cond1"
    model.set_cond(cond=cond, pdf=normal_pdf, prior_empty=prior_empty)
    assert model.conds[cond].pdf(0) == normal_pdf(0)
    assert model.conds[cond].prior_empty == prior_empty

def test_model_set_missing_cond():
    model = Model(['cond1', 'cond2'])
    prior_0 = 0.3
    cond = "cond3"
    with pytest.raises(ValueError) as exc_info:
        model.set_cond(cond=cond, prior_0=prior_0)
    assert str(exc_info.value) == "Condition cond3 not in model's set of conditions"
