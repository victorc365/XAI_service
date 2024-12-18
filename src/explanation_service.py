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

import default_values as dfv

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
        # Fix issue 
        self.cluster_model.cluster_model.cluster_centers_ = self.cluster_model.cluster_model.cluster_centers_.astype(float)
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
        # check safety
        data_df = pd.DataFrame.from_dict(data, orient='index').T
        print(f"Data df:{data_df}")
        data_df['allergy'] = data_df['allergy'].apply(lambda x: x if x in dfv.safety_allergies else dfv.safety_allergies[0])
        data_df['meal_type_y'] = data_df['meal_type_y'].apply(lambda x: x if x in dfv.safety_allergies else 'NotInformation')
        if self.rule_preprocessing is not None:
            X =  self.rule_preprocessing.transform(data_df)
        else:
            if transformer is not None:
                X =  transformer.transform(data_df)
            else:
                X = data_df.to_numpy()
        print(f"X: {X.shape}")
        if self.cluster_model is not None:
            embedding = np.array(data_df[embedding_cols].tolist(), dtype="float")
            print(f"Embedding size:{embedding.shape}")
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            cluster = self.cluster_model.predict(embedding)
            print(f"cluster: {cluster}")
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
    
    def inverse_data_preprocessing_for_bn(self, key: str, value: Any, decimals=2) -> Union[str, Any]:
        new_value = value
        if self.discretized_dict is not None:
            if key in self.discretized_dict:
                bins = self.discretized_dict[key]
                if value == 0:
                    new_value = f"{key} <= {np.round(bins[value], decimals=decimals)}"
                else:
                    new_value = f"{np.round(bins[value-1], decimals)} <= {key} < {(np.round(bins[value], decimals))}"
        return new_value
    
    def explain_decision_with_rules(self, data_array: Any) -> Union[List[str], str]:
        if self.rule_set is not None:
            prediction = self.rule_set.predict_numpy_rules(data_array, return_decision_path=True)
            print(f"Rule Prediction: {prediction}")
            return prediction
        else:
            raise ValueError("Rule set not loaded.")
        
    def predict_with_partial_information(self, evidence_dict):
        len_list = 1
        multi_explanations = []
        cpds = {}
        explanation = {}
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
            # predict 
            try:
                for i in range(len_list):
                    final_tmp = {}
                    for k in evidence_dict_internal.keys():
                        final_tmp[k] = evidence_dict_internal[k][i]
                    cpds = bn.inference.fit(self.bn_model, variables=["y_pred"], evidence=final_tmp)
                    max_idx = cpds.df["p"].idxmax()
                    max_prediction = cpds.df.loc[max_idx]
                    print(f"Predictions partial: {max_prediction.to_dict()}")
                    print("--------------------------------------------------------------------")
                    multi_explanations.append(max_prediction.to_dict())
                return multi_explanations
            except Exception as e:
                print(f"An error occurred while explaining: {traceback.format_exc()}")
                return None
        
    
    def explanation_bayesian_network(self, evidence_dict: Any) -> Union[List[str]]:
        len_list = 1
        multi_explanations = []
        cpds = {}
        explanation = {}
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
                for cause in evidence_dict_internal.keys():
                    try:
                        print(f"Cause: {cause}:")
                        # check safety 
                        if cause == 'allergy':
                            if evidence_dict_internal[cause][i] in dfv.safety_allergies:
                                evidence_dict_internal[cause][i] = evidence_dict_internal[cause][i]
                            else:
                                evidence_dict_internal[cause][i] = dfv.safety_allergies[0]
                        if cause == 'meal_type_y':
                            if evidence_dict_internal[cause][i] in dfv.meal_type_y:
                                evidence_dict_internal[cause][i] = evidence_dict_internal[cause][i]
                            else:
                                evidence_dict_internal[cause][i] = 'NotInformation'
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
                            explanation[f"{cause}={self.inverse_data_preprocessing_for_bn(cause, evidence_dict_internal[cause][i])}"] = proba
                            print(f"cpds: {proba}")
                            print("------------------------------------------")
                    except Exception as e:
                        print(f"An error occurred while explaining: {traceback.format_exc()}")
                        continue
                print(f"Explanation: {explanation}")
                multi_explanations.append(explanation.copy())
                explanation = {}
            return multi_explanations
        else:
            raise ValueError("BN model not loaded.")
        
    def generate_text_explanation_from_rule_prediction(self, rule_prediction: Tuple[List[np.array], List[Rule]]) -> str:
        explanations =[]
        prediction, rule_path = rule_prediction
        for i in range(len(prediction)):
            if prediction[i] > 0.6:
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
    
    def generate_text_explanation_from_bn_prediction(self, evidence_dict: Any, decimals: int = 2) -> str:
        raw_explanation = self.explanation_bayesian_network(evidence_dict)
        print(f"Raw explanation: {raw_explanation}")
        # generate text explanations 
        template = "I believe you would {prediction} because {text_xai}"
        text_explanations = []
        prediction_text = ""
        for raw_xai in raw_explanation:
            # sort from proba 
            if raw_xai.get("y_pred=1", None) is not None:
                prediction_text = f"like with probability {np.round(raw_xai['y_pred=1'], decimals)}"
                del raw_xai['y_pred=1']
            elif raw_xai.get("y_pred=0", None) is not None:
                if raw_xai['y_pred=0'] < 0.5:
                    prediction_text = f"dislike with probability {np.round(1.0 - raw_xai['y_pred=0'], decimals)}"
                else:
                    prediction_text = f"dislike with probability {np.round(raw_xai['y_pred=0'], decimals)}"
                del raw_xai['y_pred=0']
            sorted_xai = sorted(raw_xai.items(), key=lambda item: item[1], reverse=True)
            text_xai = []
            for t in sorted_xai:
                if t[1] > 0.55:
                    text_xai.append(f"{t[0]} with probability: {np.round(t[1], decimals)}") 
            text_xai = ", ".join(text_xai)
            text_explanations.append(template.format(prediction=prediction_text, text_xai=text_xai))
        return text_explanations
    
    def generate_partial_explanation_and_predictions(self, evidence, embedding_key: str = "embeddings"):
        #TODO: Test this method 
        # first predict y with the evidence 
        evidence_dict = evidence.copy()
        # extract cluster
        if embedding_key in evidence_dict.keys():
            if self.cluster_model is not None:
                embedding = np.array(evidence_dict[embedding_key], dtype="float64")
                print(f"Embedding shape {embedding.shape}")
                if embedding.ndim == 1:
                    embedding = embedding.reshape(1, -1)
                cluster = self.cluster_model.predict(embedding)
                print(f"cluster: {cluster}")
                del evidence_dict[embedding_key]
                evidence_dict["cluster"] = cluster[0]
        # prepare the evidence dictionary
        for key in evidence_dict.keys():
            if not isinstance(evidence_dict[key], List):
                evidence_dict[key] = [evidence_dict[key]]
        # predict y with the evidence and cluster
        y_pred_partial = self.predict_with_partial_information(evidence_dict)
        evidence_dict["y_pred"] = [int(y_pred_partial[0]["y_pred"])]
        print(f"Y partial: {y_pred_partial}")
        explanation_bayesian_network = self.generate_text_explanation_from_bn_prediction(evidence_dict=evidence_dict)
        return explanation_bayesian_network
        
    
    def generate_high_level_explanation(self, entities_list, predictions):
        # generate high level explanation using text explanations
        template = "I believe that you would {decision} because the combination of your {entities} is {high_low} compatible"
        answer = []
        for pred in predictions:
            if pred >= 0.7:
                answer.append(template.format(decision="like", entities=", ".join(entities_list), high_low="high"))
            elif pred > 0.4 and pred < 0.7:
                answer.append(template.format(decision="may like", entities=", ".join(entities_list), high_low="middle"))
            else:
                answer.append(template.format(decision="dislike", entities=", ".join(entities_list), high_low="low"))
        return answer
        
    
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
    