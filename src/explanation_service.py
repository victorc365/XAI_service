import traceback
import dill
import pickle
import bnlearn as bn
from typing import Any, Union, List, Tuple
from dexire_pro.core.clustering_explanations import ClusterAnalysis
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from recommender_service import RecommenderService
import pandas as pd
import joblib
import numpy as np
from dexire.core.rule_set import RuleSet
from dexire.core.rule import Rule
from dexire.core.clause import ConjunctiveClause, DisjunctiveClause
from dexire.core.expression import Expr

class ExplanationService:
    def __init__(self):
        self.rule_set = None
        self.bn_model = None
        self.cluster_model = None
        self.rule_preprocessing = None
        self.bn_model_preprocessing = None
        self.discretized_dict = None
        self.rule_text_xai_template = "I believe that you {prediction} because {reasoning}."
        
    def load_cluster_model(self, cluster_model_path: str) -> None:
        self.cluster_model = ClusterAnalysis()
        self.cluster_model.load_cluster_model(cluster_model_path)
        print(f"Cluster model loaded successfully!")
    
    def load_rule_set(self, rule_set_path: str) -> None:
        with open(rule_set_path, 'rb') as file:
            self.rule_set = dill.load(file)
        print(dir(self.rule_set))
        print(f"Rule set loaded successfully!")
            
    def load_bn_model(self, bn_model_path: str) -> None:
        self.bn_model = bn.load(bn_model_path)
        print(f"BN model loaded successfully!")
        
    def load_rule_set_preprocessing(self, preprocessing_path: str) -> dict:
        with open(preprocessing_path, 'rb') as f:
            self.rule_preprocessing = joblib.load(f)
            
    def load_discretized_dict(self, dict_path: str) -> None:
        with open(dict_path, 'rb') as f:
            self.discretized_dict = dill.load(f)
        print(f"Discretized dict loaded successfully!")
        
    def data_preprocessing_for_rules(self, data: dict, transformer: Any = None, embedding_cols: str="") -> dict:
        data_df = pd.DataFrame.from_dict(data, orient='index').T
        if self.rule_preprocessing is not None:
            X =  self.rule_preprocessing.transform(data_df)
        else:
            if transformer is not None:
                X =  transformer.transform(data_df)
            else:
                X = data_df.to_numpy()
        if self.cluster_model is not None:
            if data[embedding_cols].ndim == 1:
                data[embedding_cols] = data[embedding_cols].reshape(1, -1)
            cluster = self.cluster_model.predict(data[embedding_cols])
            X_final = np.column_stack((X, cluster))
        else:
            X_final = X
            print(f"X: {X.shape}, {X_final.shape} X_final")
        return X_final
    
    def data_preprocessing_for_bn(self, data: dict) -> dict:
        new_data = data.copy()
        if self.discretized_dict is not None:
            for key in data.keys():
                if key in self.discretized_dict:
                    new_data[key] = np.digitize([data[key]], bins=self.discretized_dict[key])
        for key in data.keys():
            if isinstance(new_data[key], np.ndarray):
                new_data[key] = new_data[key].flatten().tolist()
        return new_data
    
    def explain_decision_with_rules(self, data_array: Any) -> Union[List[str], str]:
        if self.rule_set is not None:
            prediction = self.rule_set.predict_numpy_rules(data_array, return_decision_path=True)
            print(f"Rule Prediction: {prediction}")
            return prediction
        else:
            raise ValueError("Rule set not loaded.")
    
    def explanation_bayesian_network(self, evidence_dict: Any) -> Union[List[str]]:
        len_list = 1
        multi_explanations = []
        cpds = {}
        nodes_to_exclude = []
        evidence_dict_internal = evidence_dict.copy()
        if self.bn_model is not None:
            for key in evidence_dict.keys():
                if key  not in self.bn_model["model"]:
                    del evidence_dict_internal[key]
                    nodes_to_exclude.append(key)
                else:
                    len_list = len(evidence_dict[key])
            print(f"Excluded nodes: {nodes_to_exclude}")
            for i in range(len_list):
                print(f"user: {1}")
                for cause in evidence_dict_internal.keys():
                    try:
                        print(f"Cause: {cause}:")
                        if cause in evidence_dict_internal.keys():
                            tem_evident_dict = evidence_dict_internal.copy()
                            print(f"Evidence: {evidence_dict_internal[cause][i]}")
                            del tem_evident_dict[cause]
                            final_temp = {}
                            for key in tem_evident_dict.keys():
                                final_temp[key] = tem_evident_dict[key][i]
                            print(f"Evidence: {final_temp}")
                            cpds[cause] = bn.inference.fit(self.bn_model, variables=[cause], evidence=final_temp)
                            proba = cpds[cause].get_value(**{cause: evidence_dict_internal[cause][i]})
                            cpds[cause] = proba
                            print(f"cpds: {proba}")
                            print("------------------------------------------")
                            multi_explanations.append(cpds.copy())
                    except Exception as e:
                        print(f"An error occurred while explaining: {traceback.format_exc()}")
                        continue
            return multi_explanations
        else:
            raise ValueError("BN model not loaded.")
        
    def generate_text_explanation_from_rule_prediction(self, rule_prediction: Tuple[List[np.array], List[Rule]]) -> str:
        explanations =[]
        prediction, rule_path = rule_prediction
        for i in range(len(prediction)):
            if prediction[i] == 1:
                text_pred = "like"
            else:
                text_pred = "dislike"
            reasoning = rule_path[i]
            reasoning_text = ""
            for rule in reasoning:
                premise = rule.premise
                reasoning_text += f"{premise}"
            explanation = self.rule_text_xai_template.format(prediction=text_pred, reasoning=reasoning_text)
            explanations.append(explanation)
        return explanations
    
    
    
if __name__ == "__main__":
    recommender_service = RecommenderService("model_assets/model_full_use__fold_0.tf")
    model_inputs_and_type = recommender_service.get_model_inputs_and_type()
    print(model_inputs_and_type)
    recommender_service.load_embeddings("model_assets/full_recipe_embedding_BERT_v2_17_may_recipeId.npz")
    # example 
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
     'embeddings': recommender_service.get_embedding_for_recipe_id("food_0").reshape(1, -1), 
     'ethnicity': ['White'], 
     'fat': [30], 
     'fiber': [29], 
     'height': [165], 
     'life_style': ['Sedentary'], 
     'marital_status': ['Single'], 
     'meal_type_y': ['lunch'], 
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
    recommended_items = recommender_service.recommend_items(sample)
    print(recommended_items)
    columns_dict = recommender_service.get_model_data_types()
    print(columns_dict)
    # Explanation
    explanation_service = ExplanationService()
    explanation_service.load_cluster_model('model_assets/new_experiment_bert_cluster_full_model.pkl')
    explanation_service.load_rule_set('model_assets/new_experiments_ruleset_bert_0.pkl')
    explanation_service.load_bn_model('model_assets/bn_model_bert.pkl')
    explanation_service.load_rule_set_preprocessing('model_assets/preprocessor_rule.pkl')
    explanation_service.load_discretized_dict('model_assets/discretize_dict.pkl')
    # transform sample 
    X_final = explanation_service.data_preprocessing_for_rules(sample, embedding_cols='embeddings')
    #X_final = np.repeat(X_final, 4, axis=0)
    print()
    print(X_final.shape)
    rule_prediction = explanation_service.explain_decision_with_rules(X_final)
    print(f"Explanation: {rule_prediction}")
    expa = explanation_service.generate_text_explanation_from_rule_prediction(rule_prediction)
    print(f"Text Explanation: {expa}")
    processed_sample =sample.copy()
    processed_sample.update({'y_pred': rule_prediction[0]})
    post_processed_sample = explanation_service.data_preprocessing_for_bn(processed_sample)
    print(f"Post: {post_processed_sample}")
    expa_bn = explanation_service.explanation_bayesian_network(post_processed_sample)
    print(f"Bayesian Network Explanation: {expa_bn}")
    