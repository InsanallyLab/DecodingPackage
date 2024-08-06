import pytest
import numpy as np
from decoder.core.ndecoder import NDecoder
from decoder.core.model import Model
from pytest_mock import mocker 

### Initialization tests ###
def test_ndecoder_init():
    bw = 0.1
    min_ISIs = 10
    conditions = ['cond1']
    decoder = NDecoder(bw=bw, min_ISIs=min_ISIs, conditions=conditions)
    
    assert(decoder.bw == bw)
    assert(decoder.min_ISIs == min_ISIs)
    assert(decoder.conditions == conditions)
    assert(isinstance(decoder.model, Model))

def test_ndecoder_init_missing_args():
    min_ISIs = 10
    conditions = ['cond1']
    
    with pytest.raises(TypeError):
        decoder = NDecoder(min_ISIs=min_ISIs, conditions=conditions)

### _create_condition_subset_mapping tests ###
def test_ndecoder_subset_mapping():
    bw = 0.1
    min_ISIs = 10
    conditions = ['cond1']
    decoder = NDecoder(bw=bw, min_ISIs=min_ISIs, conditions=conditions)
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array(['cond1', 'cond1', 'cond1'])

    mapping = decoder._create_condition_subset_mapping(X, y)
    assert len(mapping['cond1']) == 3
    assert all(x in mapping['cond1'] for x in X)

def test_ndecoder_subset_mapping_many_conds():
    bw = 0.1
    min_ISIs = 10
    conditions = ['cond1', 'cond2', 'cond3']
    decoder = NDecoder(bw=bw, min_ISIs=min_ISIs, conditions=conditions)
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array(['cond1', 'cond2', 'cond1'])

    mapping = decoder._create_condition_subset_mapping(X, y)
    assert len(mapping['cond1']) == 2
    assert X[0] in mapping['cond1']
    assert X[2] in mapping['cond1']
    assert len(mapping['cond2']) == 1
    assert X[1] in mapping['cond2']
    assert len(mapping['cond3']) == 0

def test_ndecoder_subset_mapping_empty_inp():
    bw = 0.1
    min_ISIs = 10
    conditions = ['cond1', 'cond2']
    decoder = NDecoder(bw=bw, min_ISIs=min_ISIs, conditions=conditions)
    X = np.array([])
    y = np.array([])

    mapping = decoder._create_condition_subset_mapping(X, y)
    assert len(mapping['cond1']) == 0
    assert len(mapping['cond2']) == 0

def test_ndecoder_subset_mapping_invalid_cond():
    bw = 0.1
    min_ISIs = 10
    conditions = ['cond1', 'cond2']
    decoder = NDecoder(bw=bw, min_ISIs=min_ISIs, conditions=conditions)
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array(['cond1', 'cond3', 'cond2'])

    with pytest.raises(ValueError) as exc_info:
        mapping = decoder._create_condition_subset_mapping(X, y)
    assert str(exc_info.value) == "Invalid labels found: cond3"


### estimate_ISI_distribution tests ###
def test_ndecoder_estimate_ISI_distribution_pdf(mocker):
    bw = 0.1
    min_ISIs = 10
    conditions = ['cond1']
    decoder = NDecoder(bw=bw, min_ISIs=min_ISIs, conditions=conditions)

    mock_fftkde_eval = mocker.patch("decoder.core.ndecoder.FFTKDE.evaluate")
    mock_fftkde_eval.return_value = [2.89 for i in range(100)]

    log_ISIs = [2.4, 4.8, 3.2]
    pdf_interpolator, _ = decoder.estimate_ISI_distribution(
        log_ISIs, 
        decoder.bw)

    assert pdf_interpolator(3) == 2.89
    assert pdf_interpolator(-1.9) == 2.89
    assert pdf_interpolator(19.2) == 2.89

def test_ndecoder_estimate_ISI_distribution_pdf_inv_cdf(mocker):
    bw = 0.1
    min_ISIs = 10
    conditions = ['cond1']
    decoder = NDecoder(bw=bw, min_ISIs=min_ISIs, conditions=conditions)

    mock_fftkde_eval = mocker.patch("decoder.core.ndecoder.FFTKDE.evaluate")
    mock_fftkde_eval.return_value = [2.89 for i in range(100)]

    mock_interp1d = mocker.patch("decoder.core.ndecoder.interp1d")
    mock_interp1d.return_value = lambda x : 0.84

    log_ISIs = [2.4, 4.8, 3.2]
    pdf_interpolator, cdf_interpolator = decoder.estimate_ISI_distribution(
        log_ISIs, 
        decoder.bw)

    assert pdf_interpolator(3) == 0.84
    assert pdf_interpolator(-1.9) == 0.84
    assert cdf_interpolator(0.4) == 0.84
    assert cdf_interpolator(2.54) == 0.84

### fit tests ###
def test_ndecoder_fit(mocker):
    bw = 0.1
    min_ISIs = 1
    conditions = ['cond1', 'cond2']
    decoder = NDecoder(bw=bw, min_ISIs=min_ISIs, conditions=conditions)

    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array(['cond1', 'cond2', 'cond1'])

    mock_fftkde_eval = mocker.patch("decoder.core.ndecoder.FFTKDE.evaluate")
    mock_fftkde_eval.return_value = [2.89 for i in range(100)]

    mock_interp1d = mocker.patch("decoder.core.ndecoder.interp1d")
    mock_interp1d.return_value = lambda x : 0.1 * x

    decoder.fit(X, y)
    model = decoder.model
    assert model.all.pdf(3) == 0.1*3
    assert model.all.pdf(-1.9) == 0.1*-1.9
    assert model.all.inv_cdf(0.4) == 0.1*0.4
    assert model.all.inv_cdf(0.65) == 0.1*0.65
    assert model.all.prior_0 == None
    assert model.all.prior_empty == None

    assert model.conds['cond1'].pdf(3) == 0.1*3
    assert model.conds['cond2'].pdf(-1.9) == 0.1*-1.9
    assert model.conds['cond1'].inv_cdf(0.4) == 0.1*0.4
    assert model.conds['cond2'].inv_cdf(0.65) == 0.1*0.65

    assert model.conds['cond1'].prior_0 == 0.5
    assert model.conds['cond2'].prior_0 == 0.5
    assert model.conds['cond1'].prior_empty == 0.5
    assert model.conds['cond2'].prior_empty == 0.5

def test_ndecoder_fit_insufficient_ISIs(mocker):
    bw = 0.1
    min_ISIs = 4
    conditions = ['cond1', 'cond2']
    decoder = NDecoder(bw=bw, min_ISIs=min_ISIs, conditions=conditions)

    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array(['cond1', 'cond2', 'cond1'])

    mock_fftkde_eval = mocker.patch("decoder.core.ndecoder.FFTKDE.evaluate")
    mock_fftkde_eval.return_value = [2.89 for i in range(100)]

    mock_interp1d = mocker.patch("decoder.core.ndecoder.interp1d")
    mock_interp1d.return_value = lambda x : 0.1 * x

    decoder.fit(X, y)
    assert decoder.model.all.pdf == None
    assert decoder.model.all.inv_cdf == None
    assert decoder.model.all.prior_0 == None
    assert decoder.model.all.prior_empty == None
    for cond in conditions:
        assert decoder.model.conds[cond].pdf == None
        assert decoder.model.conds[cond].inv_cdf == None
        assert decoder.model.conds[cond].prior_0 == None
        assert decoder.model.conds[cond].prior_empty == None
    
def test_ndecoder_fit_empty_ISIs(mocker):
    bw = 0.1
    min_ISIs = 1
    conditions = ['cond1', 'cond2']
    decoder = NDecoder(bw=bw, min_ISIs=min_ISIs, conditions=conditions)

    X = np.array([[], [4, 5, 6], [7, 8, 9]], dtype=object)
    y = np.array(['cond1', 'cond2', 'cond1'])

    mock_fftkde_eval = mocker.patch("decoder.core.ndecoder.FFTKDE.evaluate")
    mock_fftkde_eval.return_value = [2.89 for i in range(100)]

    mock_interp1d = mocker.patch("decoder.core.ndecoder.interp1d")
    mock_interp1d.return_value = lambda x : 0.1 * x

    decoder.fit(X, y)
    model = decoder.model
    assert model.all.pdf(3) == 0.1*3
    assert model.all.pdf(-1.9) == 0.1*-1.9
    assert model.all.inv_cdf(0.4) == 0.1*0.4
    assert model.all.inv_cdf(0.65) == 0.1*0.65
    assert model.all.prior_0 == None
    assert model.all.prior_empty == None

    assert model.conds['cond1'].pdf(3) == 0.1*3
    assert model.conds['cond2'].pdf(-1.9) == 0.1*-1.9
    assert model.conds['cond1'].inv_cdf(0.4) == 0.1*0.4
    assert model.conds['cond2'].inv_cdf(0.65) == 0.1*0.65

    assert model.conds['cond1'].prior_0 == 0.5
    assert model.conds['cond2'].prior_0 == 0.5
    assert model.conds['cond1'].prior_empty == 2/3
    assert model.conds['cond2'].prior_empty == 1/3

### _predict_single_trial tests ###
def test_predict_single_trial_empty_ISIs():
    bw = 0.1
    min_ISIs = 1
    conditions = ['cond1', 'cond2']
    n_decoder = NDecoder(bw=bw, min_ISIs=min_ISIs, conditions=conditions)

    model = Model(conditions)
    model.set_cond(cond = 'cond1', prior_0 = 0.5, prior_empty= 0.6)
    model.set_cond(cond = 'cond2', prior_0 = 0.5, prior_empty = 0.4)

    n_decoder.model = model
    trialISIs = []
    
    max_cond, prob_max_cond, all_probs, if_ISIs_empty = n_decoder._predict_single_trial(trialISIs)

    assert max_cond == 'cond1'
    assert prob_max_cond == 0.6
    assert all_probs['cond1'] == 0.6
    assert all_probs['cond2'] == 0.4
    assert if_ISIs_empty == True

def test_predict_single_trial_with_ISIs():
    bw = 0.1
    min_ISIs = 1
    conditions = ['cond1', 'cond2']
    n_decoder = NDecoder(bw=bw, min_ISIs=min_ISIs, conditions=conditions)

    model = Model(conditions)
    model.set_cond(cond = 'cond1', pdf = lambda x: 0.1 * x, prior_0 = 0.5, prior_empty = 0.6)
    model.set_cond(cond = 'cond2', pdf = lambda x: 0.2 * x, prior_0 = 0.5, prior_empty = 0.4)

    n_decoder.model = model
    trialISIs = [1, 2, 3]
    
    max_cond, prob_max_cond, all_probs, if_ISIs_empty = n_decoder._predict_single_trial(trialISIs)
    
    assert max_cond == 'cond2'
    assert np.isclose(prob_max_cond, 0.888888)
    assert np.isclose(all_probs['cond1'], 0.111111)
    assert np.isclose(all_probs['cond2'], 0.888888)
    assert if_ISIs_empty == False

def test_predict_single_trial_equal_probabilities():
    bw = 0.1
    min_ISIs = 1
    conditions = ['cond1', 'cond2']
    n_decoder = NDecoder(bw=bw, min_ISIs=min_ISIs, conditions=conditions)

    model = Model(conditions)
    model.set_cond(cond = 'cond1', pdf = lambda x: 0.2 * x, prior_0 = 0.5, prior_empty = 0.6)
    model.set_cond(cond = 'cond2', pdf = lambda x: 0.2 * x, prior_0 = 0.5, prior_empty = 0.4)

    n_decoder.model = model
    trialISIs = [1, 2, 3]

    max_cond, prob_max_cond, all_probs, if_ISIs_empty  = n_decoder._predict_single_trial(trialISIs)
    
    assert max_cond in ['cond1', 'cond2']
    assert prob_max_cond == 0.5
    assert all_probs['cond1'] == 0.5
    assert all_probs['cond2'] == 0.5
    assert if_ISIs_empty == False 


### predict_conditions tests ###
def test_predict_conditions_no_trials():
    bw = 0.1
    min_ISIs = 1
    conditions = ['cond1', 'cond2']
    n_decoder = NDecoder(bw=bw, min_ISIs=min_ISIs, conditions=conditions)

    model = Model(conditions)
    model.set_cond(cond = 'cond1', pdf = lambda x: 0.1 * x, prior_0 = 0.5, prior_empty = 0.6)
    model.set_cond(cond = 'cond2', pdf = lambda x: 0.2 * x, prior_0 = 0.5, prior_empty = 0.4)

    n_decoder.model = model
    trialISIs = []

    predicted_conditions = n_decoder.predict_conditions(trialISIs)
    assert np.array_equal(predicted_conditions, [])

def test_predict_conditions():
    bw = 0.1
    min_ISIs = 1
    conditions = ['cond1', 'cond2']
    n_decoder = NDecoder(bw=bw, min_ISIs=min_ISIs, conditions=conditions)

    model = Model(conditions)
    model.set_cond(
        cond = 'cond1', 
        pdf = lambda X : [0.1 * x if x < 4 else 0.2 * x for x in X], 
        prior_0 = 0.5, 
        prior_empty = 0.6)
    model.set_cond(
        cond = 'cond2', 
        pdf = lambda X : [0.2 * x if x < 4 else 0.1 * x for x in X], 
        prior_0 = 0.5, 
        prior_empty = 0.4)

    n_decoder.model = model
    trialISIs = [[1, 2, 3], [4, 5, 6]]

    predicted_conditions = n_decoder.predict_conditions(trialISIs)
    assert np.array_equal(predicted_conditions, ['cond2', 'cond1'])


### predict_condition_probs tests ###
def test_predict_condition_probs_no_trials():
    bw = 0.1
    min_ISIs = 1
    conditions = ['cond1', 'cond2']
    n_decoder = NDecoder(bw=bw, min_ISIs=min_ISIs, conditions=conditions)

    model = Model(conditions)
    model.set_cond(cond = 'cond1', pdf = lambda x: 0.1 * x, prior_0 = 0.5, prior_empty = 0.6)
    model.set_cond(cond = 'cond2', pdf = lambda x: 0.2 * x, prior_0 = 0.5, prior_empty = 0.4)

    n_decoder.model = model
    trialISIs = []

    condition_probs_df = n_decoder.predict_condition_probs(trialISIs)
    assert condition_probs_df.empty 
    assert np.array_equal(condition_probs_df.columns, conditions)

def test_predict_condition_probs():
    bw = 0.1
    min_ISIs = 1
    conditions = ['cond1', 'cond2']
    n_decoder = NDecoder(bw=bw, min_ISIs=min_ISIs, conditions=conditions)

    model = Model(conditions)
    model.set_cond(
        cond = 'cond1', 
        pdf = lambda X : [0.1 * x if x < 4 else 0.2 * x for x in X], 
        prior_0 = 0.5, 
        prior_empty = 0.6)
    model.set_cond(
        cond = 'cond2', 
        pdf = lambda X : [0.2 * x if x < 4 else 0.1 * x for x in X], 
        prior_0 = 0.5, 
        prior_empty = 0.4)

    n_decoder.model = model
    trialISIs = [[1, 2, 3], [4, 5, 6]]

    condition_probs_df = n_decoder.predict_condition_probs(trialISIs)
    assert np.array_equal(condition_probs_df.columns, conditions)
    assert np.isclose(condition_probs_df['cond1'][0], 0.111111)
    assert np.isclose(condition_probs_df['cond2'][0], 0.888888)
    assert np.isclose(condition_probs_df['cond1'][1], 0.888888)
    assert np.isclose(condition_probs_df['cond2'][1], 0.111111)


### generate_stratified_folds tests ###
def test_generate_strat_folds_empty():
    X = []
    y = []
    K = 2

    with pytest.raises(ValueError) as exc_info:
        train_validate_pairs = NDecoder.generate_stratified_K_folds(X, y, K)
    assert str(exc_info.value) == "Found array with 0 sample(s) (shape=(0,)) while a minimum of 1 is required."

def test_generate_strat_folds():
    X = [[1, 2, 3], [4, 5, 6], [1, 2, 3], [4, 5, 6]]
    y = ["cond2", "cond1", "cond2", "cond1"]
    K = 2

    train_validate_pairs = NDecoder.generate_stratified_K_folds(X, y, K)
    assert len(train_validate_pairs) == K
    (train_X, train_y), (validate_X, validate_y )= train_validate_pairs[0]
    assert len(train_X) == len(train_y) == len(X) // 2
    assert len(validate_X) == len(validate_y) == len(X) // 2
    (train_X, train_y), (validate_X, validate_y )= train_validate_pairs[1]
    assert len(train_X) == len(train_y) == len(X) // 2
    assert len(validate_X) == len(validate_y) == len(X) // 2


### calculate_accuracy tests ###
def test_calculate_accuracy_no_trials():
    bw = 0.1
    min_ISIs = 1
    conditions = ['cond1', 'cond2']
    n_decoder = NDecoder(bw=bw, min_ISIs=min_ISIs, conditions=conditions)

    model = Model(conditions)
    model.set_cond(cond = 'cond1', pdf = lambda x: 0.1 * x, prior_0 = 0.5, prior_empty = 0.6)
    model.set_cond(cond = 'cond2', pdf = lambda x: 0.2 * x, prior_0 = 0.5, prior_empty = 0.4)

    n_decoder.model = model
    test_X = []
    test_y = []

    accuracy, frac_empty_ISIs = n_decoder.calculate_accuracy(test_X=test_X, test_y=test_y)
    assert np.isnan(accuracy)
    assert np.isnan(frac_empty_ISIs)

def test_calculate_accuracy_empty_ISIs():
    bw = 0.1
    min_ISIs = 1
    conditions = ['cond1', 'cond2']
    n_decoder = NDecoder(bw=bw, min_ISIs=min_ISIs, conditions=conditions)

    model = Model(conditions)
    model.set_cond(cond = 'cond1', pdf = lambda x: 0.1 * x, prior_0 = 0.5, prior_empty = 0.6)
    model.set_cond(cond = 'cond2', pdf = lambda x: 0.2 * x, prior_0 = 0.5, prior_empty = 0.4)

    n_decoder.model = model
    test_X = [[]]
    test_y = ["cond2"]

    accuracy, frac_empty_ISIs = n_decoder.calculate_accuracy(test_X=test_X, test_y=test_y)
    assert accuracy == 0
    assert frac_empty_ISIs == 1

def test_calculate_accuracy():
    bw = 0.1
    min_ISIs = 1
    conditions = ['cond1', 'cond2']
    n_decoder = NDecoder(bw=bw, min_ISIs=min_ISIs, conditions=conditions)

    model = Model(conditions)
    model.set_cond(
        cond = 'cond1', 
        pdf = lambda X : [0.1 * x if x < 4 else 0.2 * x for x in X], 
        prior_0 = 0.5, 
        prior_empty = 0.6)
    model.set_cond(
        cond = 'cond2', 
        pdf = lambda X : [0.2 * x if x < 4 else 0.1 * x for x in X], 
        prior_0 = 0.5, 
        prior_empty = 0.4)

    n_decoder.model = model

    test_X = [[1, 2, 3], [4, 5, 6]]
    test_y = ["cond2", "cond2"]

    accuracy, frac_empty_ISIs = n_decoder.calculate_accuracy(test_X=test_X, test_y=test_y)
    assert accuracy == 0.5
    assert frac_empty_ISIs == 0