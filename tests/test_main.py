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


