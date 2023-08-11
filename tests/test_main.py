import pytest
import numpy as np
from decoder.main import * # Make sure you import the classes correctly
from pytest_mock import mocker 

@pytest.fixture
def sample_model():
    model = Model(['condition1', 'condition2'])  # Replace with your actual conditions
    return model

def test_model_init(sample_model):
    assert len(sample_model.conds) == 2
    assert isinstance(sample_model.all, Likelihood)

def test_likelihood_init():
    likelihood = Likelihood()
    assert likelihood.pdf is None
    assert likelihood.inv_cdf is None
    assert likelihood.prior_0 is None
    assert likelihood.prior_empty is None

def test_likelihood_set_val():
    likelihood = Likelihood()
    likelihood.set_val(pdf=[1, 2, 3], inv_cdf=[0.1, 0.2, 0.3])
    assert likelihood.pdf == [1, 2, 3]
    assert likelihood.inv_cdf == [0.1, 0.2, 0.3]

def test_ndeocoder_create_condition_subset_mapping():
    # Test case 1: Single condition
    decoder = NDecoder(bandwidth=0.1, min_isis=10, conditions=['condition1'])
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array(['condition1', 'condition1', 'condition1'])

    mapping = decoder._create_condition_subset_mapping(X, y)
    assert len(mapping['condition1']) == 3

    # Test case 2: Multiple conditions
    decoder = NDecoder(bandwidth=0.1, min_isis=10, conditions=['condition1', 'condition2'])
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array(['condition1', 'condition2', 'condition1'])

    mapping = decoder._create_condition_subset_mapping(X, y)
    assert len(mapping['condition1']) == 2
    assert len(mapping['condition2']) == 1

    # Test case 3: Empty input
    decoder = NDecoder(bandwidth=0.1, min_isis=10, conditions=['condition1', 'condition2'])
    X = np.array([])
    y = np.array([])

    mapping = decoder._create_condition_subset_mapping(X, y)
    assert len(mapping['condition1']) == 0
    assert len(mapping['condition2']) == 0

    # Test case 4: No matching condition
    decoder = NDecoder(bandwidth=0.1, min_isis=10, conditions=['condition1', 'condition2'])
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array(['condition1', 'condition3', 'condition2'])

    with pytest.raises(ValueError) as exc_info:
        mapping = decoder._create_condition_subset_mapping(X, y)
    assert str(exc_info.value) == "Invalid labels found: condition3"

    # Test case 5: Mixed conditions
    decoder = NDecoder(bandwidth=0.1, min_isis=10, conditions=['condition1', 'condition2', 'condition3'])
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array(['condition1', 'condition2', 'condition1'])

    mapping = decoder._create_condition_subset_mapping(X, y)
    assert len(mapping['condition1']) == 2
    assert len(mapping['condition2']) == 1
    assert len(mapping['condition3']) == 0 

def test_predict_single_trial_empty_isi():
    conditions = ['A', 'B']
    n_decoder = NDecoder(bandwidth=10, min_isis=5, conditions=conditions)
    model = Model(conditions)
    model.set_all(10, 15)
    model.set_cond('A', pdf = None, inv_cdf = None, prior_0 = 0.5, prior_empty= 0.6)
    model.set_cond('B', pdf = None, inv_cdf = None, prior_0 = 0.5, prior_empty = 0.4)

    n_decoder.model = model
    trialISIs = []
    
    result = n_decoder._predict_single_trial(model, trialISIs)
    
    assert result == ('A', 0.6, True), "Prediction with empty ISIs failed"

def test_predict_single_trial_non_empty_isi():
    conditions = ['A', 'B']
    n_decoder = NDecoder(bandwidth=10, min_isis=5, conditions=conditions)
    model = Model(conditions)
    model.set_all(10, 15)
    model.set_cond('A', lambda x: 0.1 * x, 20, 0.5, 0.6)
    model.set_cond('B', lambda x: 0.2 * x, 25, 0.5, 0.4)

    n_decoder.model = model
    trialISIs = [1, 2, 3]
    
    result = n_decoder._predict_single_trial(model, trialISIs)
    
    assert result[0] == 'B', "Prediction with non-empty ISIs failed"
    assert np.isclose(result[1], 0.888888888), "Probability calculation failed"
    assert result[2] == False, "ISI empty flag failed"


def test_predict_single_trial_with_same_probability():
    conditions = ['A', 'B']
    n_decoder = NDecoder(bandwidth=10, min_isis=5, conditions=conditions)
    model = Model(conditions)
    model.set_all(10, 15)
    model.set_cond('A', lambda x: 0.2 * x, 20, 0.5, 0.6)
    model.set_cond('B', lambda x: 0.2 * x, 25, 0.5, 0.4)

    n_decoder.model = model
    trialISIs = [1, 2, 3]

    result = n_decoder._predict_single_trial(model, trialISIs)
    
    assert result[0] in ['A', 'B'], "Prediction with equal probabilities failed"


