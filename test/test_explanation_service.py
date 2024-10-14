import sys 
import os 
import pytest
from pathlib import Path
import pandas as pd 
import numpy as np 
from typing import List

# add modules to system path 
current_file_parent = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.path.join(current_file_parent, 'src'))
# import testing modules 
from explanation_service import ExplanationService

@pytest.fixture 
def explanation_service():
    explanation_service = ExplanationService()
    cluster_path = os.path.join(current_file_parent,
                            "model_assets",
                            "new_experiment_complex_model_bert_cluster_full_model.pkl")
    explanation_service.load_cluster_model(cluster_path)
    path_to_ruleset = os.path.join(current_file_parent,
                                "model_assets",
                                "new_experiments_ruleset_bert_0_Full_model.pkl")
    explanation_service.load_rule_set(path_to_ruleset)
    path_preprocessing_ruleset = os.path.join(current_file_parent,
                                        "model_assets",
                                        "preprocessor_rules_new_model_bert.pkl")
    explanation_service.load_rule_set_preprocessing(path_preprocessing_ruleset)
    path_to_bn_learn_model = os.path.join(current_file_parent,
                                "model_assets",
                                "bn_learn_model_bert_0_Full_model.pkl")
    explanation_service.load_bn_model(path_to_bn_learn_model)
    path_to_bn_learn_preprocessor = os.path.join(current_file_parent,
                                                "model_assets",
                                                "preprocessor_bn_bert_0_Full_model.pkl")
    explanation_service.load_bn_preprocessing(path_to_bn_learn_preprocessor)
    return explanation_service

@pytest.fixture
def load_explanation_data():
    # load embedding dict 
    path_to_embedding = os.path.join(current_file_parent, "model_assets", "full_recipe_embedding_BERT_v2_17_may_recipeId.npz")
    embedding_dict = dict(np.load(path_to_embedding))
    sample = {'BMI': ['healthy'], 
     'age_range': ['30-39'], 
     'allergens': ["tree nuts"], 
     'allergy': ["soy"], 
     'calories': [650], 
     'carbohydrates': [39], 
     'clinical_gender': ['M'], 
     'cultural_factor': ['vegan_observant'], 
     'cultural_restriction': ['vegetarian'], 
     'current_daily_calories': [1700], 
     'current_working_status': ['Unemployed'],  
     'day_number': [0], 
     'embeddings': embedding_dict.get("food_0", None).reshape(1, -1), 
     'ethnicity': ['White'], 
     'fat': [30], 
     'fiber': [29], 
     'height': [165], 
     'life_style': ['Sedentary'], 
     'marital_status': ['Single'], 
     'meal_type_x': ['lunch'], 
     'next_BMI': ['healthy'],  
     'nutrition_goal': ['maintain_fit'], 
     'place_of_meal_consumption': ['home'], 
     'price': [2], 
     'projected_daily_calories': [2200], 
     'protein': [30], 
     'social_situation_of_meal_consumption': ['alone'], 
     'taste': ['sweet'], 
     'time_of_meal_consumption': [12.01], 
     'weight': [65]}
    return sample

@pytest.fixture
def load_partial_data():
    sample = {'BMI': ['healthy'], 
     'age_range': ['30-39'], 
     'allergens': ["tree nuts"], 
     'allergy': ["soy"], 
     'calories': [650], 
     'carbohydrates': [39], 
     'clinical_gender': ['M'], 
     'cultural_factor': ['vegan_observant'], 
     'cultural_restriction': ['vegetarian'], 
     'current_daily_calories': [1700], 
     'current_working_status': ['Unemployed'],  
     'ethnicity': ['White'], 
     'fat': [30], 
     'fiber': [29], 
     'height': [165], 
     'life_style': ['Sedentary'], 
     'marital_status': ['Single'], 
     'meal_type_x': ['lunch'], 
     'nutrition_goal': ['maintain_fit'], 
     'place_of_meal_consumption': ['home'], 
     'price': [2], 
     'projected_daily_calories': [2200], 
     'protein': [30], 
     'social_situation_of_meal_consumption': ['alone'], 
     'taste': ['sweet'], 
     'time_of_meal_consumption': [12.01], 
     'weight': [65]}
    return sample
    
    
def test_data_preprocessing_for_rules(explanation_service, load_explanation_data):
    sample = load_explanation_data.copy()
    X_final = explanation_service.data_preprocessing_for_rules(sample, embedding_cols='embeddings')
    print(f"Transformed data: {X_final.shape}")
    assert X_final is not None 
    
    
def test_data_preprocessing_for_bn_with_pipeline(explanation_service, load_explanation_data):
    sample = load_explanation_data.copy()
    sample.update({'y_pred': [0]})
    X_final = explanation_service.data_preprocessing_for_bn_with_pipeline(sample, embedding_cols='embeddings')
    print(f"Transformed data: {X_final.shape}")
    assert X_final is not None
    
def test_explain_decision_with_rules(explanation_service, load_explanation_data):
    sample = load_explanation_data.copy()
    X_final = explanation_service.data_preprocessing_for_rules(sample, embedding_cols='embeddings')
    print(f"Transformed data: {X_final.shape}")
    print(f"Type of the object: {type(X_final)}")
    rule_prediction = explanation_service.explain_decision_with_rules(data_array = X_final)
    print(f"Rule prediction: {rule_prediction}")
    assert rule_prediction is not None
    
def test_partial_bn_explanation_transform(explanation_service, load_partial_data):
    data = load_partial_data.copy()
    evidence_dict = explanation_service.partial_bn_explanation_transform(data_dict=data)
    print(f"Evidence dict: {evidence_dict}")
    assert evidence_dict is not None and len(evidence_dict) > 1    
    
def test_predict_with_partial_information(explanation_service, load_partial_data):
    evidence_dict = load_partial_data.copy()
    # extract cluster
    evidence_dict = explanation_service.partial_bn_explanation_transform(evidence_dict)
    print(f"{evidence_dict.keys()} dict before")
    # prepare the evidence dictionary
    for key in evidence_dict.keys():
        if not isinstance(evidence_dict[key], List):
            evidence_dict[key] = [evidence_dict[key]]
    # predict y with the evidence and cluster
    print(f"{evidence_dict.keys()} dict before prediction")
    y_pred_partial = explanation_service.predict_with_partial_information(evidence_dict.copy())
    evidence_dict["identity__y_pred"] = [int(y_pred_partial[0]["identity__y_pred"])]
    print(f"Y partial: {y_pred_partial}")
    assert y_pred_partial is not None and len(y_pred_partial) > 0
    
def test_explanation_bayesian_network(explanation_service, load_explanation_data):
    sample = load_explanation_data.copy()
    sample.update({"y_pred": [0]})
    explanation = explanation_service.explanation_bayesian_network(sample)
    assert explanation is not None

def test_generate_text_explanation_from_rule_prediction(explanation_service, load_explanation_data):
    sample = load_explanation_data.copy()
    X_final = explanation_service.data_preprocessing_for_rules(sample, embedding_cols='embeddings')
    print(f"Transformed data: {X_final.shape}")
    rule_prediction = explanation_service.explain_decision_with_rules(data_array = X_final)
    answers = explanation_service.generate_text_explanation_from_rule_prediction(rule_prediction)
    print(f"Answer: {answers}")
    assert answers is not None and len(answers) > 0
    
def test_generate_text_explanation_from_bn_prediction(explanation_service, load_explanation_data):
    data = load_explanation_data.copy()
    data.update({"y_pred": [0]})
    ans = explanation_service.generate_text_explanation_from_bn_prediction(data)
    print(f"Answer: {ans}")
    assert ans is not None and len(ans) > 0

def test_generate_partial_explanation_and_predictions(explanation_service, load_partial_data):
    sample = load_partial_data.copy()
    sample.update({"y_pred": [0]})
    ans = explanation_service.generate_partial_explanation_and_predictions(sample)
    print(f"Answer: {ans}")
    assert ans is not None and len(ans) > 0

def test_generate_high_level_explanation(explanation_service):
    answer = explanation_service.generate_high_level_explanation(["profile", "context", "recipe"], [0.4, 0.8])
    print(f"Answer: {answer}")
    assert answer is not None and len(answer) > 0