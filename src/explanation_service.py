import traceback
import os 
import sys
import pathlib as pth
import dill
import pickle
import bnlearn as bn
from typing import Any, Union, List, Tuple, Dict
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


import re
def replace_bad_characters(text: str):
  new_text = text.lower()
  new_text = new_text.replace('.', '')
  new_text = re.sub(' +', '_', new_text)
  new_text = new_text.replace(';', '_')
  new_text = new_text.replace(',', '_')
  new_text = new_text.replace(' ', '_')
  new_text = new_text.replace('/', '_')
  new_text = new_text.replace('-', '_')
  #new_text = text.translate(str.maketrans('', '', string.punctuation))
  return new_text


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
            self.rule_preprocessing = dill.load(f)
            
    def load_discretized_dict(self, dict_path: str) -> None:
        # Only maintained for backwards compatibility
        with open(dict_path, 'rb') as f:
            self.discretized_dict = dill.load(f)
        print(f"Discretized dict loaded successfully!")
        
    def load_bn_preprocessing(self, preprocessing_path: str) -> dict:
        #TODO: test this module
        with open(preprocessing_path, 'rb') as f:
            self.bn_model_preprocessing = dill.load(f)
        
    def data_preprocessing_for_rules(self, data: dict, transformer: Any = None, embedding_cols: str="") -> dict:
        # check safety
        data_df = pd.DataFrame.from_dict(data, orient='index').T
        print(f"Data df:{data_df}")
        data_df['allergy'] = data_df['allergy'].apply(lambda x: x if x in dfv.safety_allergies else dfv.safety_allergies[0])
        #data_df['meal_type_y'] = data_df['meal_type_y'].apply(lambda x: x if x in dfv.safety_allergies else 'NotInformation')
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
    
    def data_preprocessing_for_bn_with_pipeline(self, 
                                                data: Dict[str, List[Any]], 
                                                embedding_cols="embeddings"):
        data_df = pd.DataFrame.from_dict(data, orient='index').T
        print(f"Data df:{data_df}")
        print(f"Data frame shape: {data_df.shape}")
        embedding = np.array(data_df[embedding_cols].tolist(), dtype="float")
        print(f"Embedding size:{embedding.shape}")
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        cluster = self.cluster_model.predict(embedding)
        print(f"cluster: {cluster}")
        data_df["cluster"] = cluster
        #TODO: check security
        x_out = None
        if self.bn_model_preprocessing is not None:
            x_out = self.bn_model_preprocessing.transform(data_df)
            feature_names = self.bn_model_preprocessing.get_feature_names_out()
            transformed_feature_names = [replace_bad_characters(feat) for feat in feature_names]
            x_out_df = pd.DataFrame(data=x_out, columns=transformed_feature_names)
            return x_out_df
        else:
            print(f"missing transformed pipeline")
            return data
    
    def data_preprocessing_for_bn(self, data: dict) -> dict:
        # old only preserved for backwards compatibility
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
        #TODO: update for the new transformers 
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
                    cpds = bn.inference.fit(self.bn_model, variables=["identity__y_pred"], evidence=final_tmp)
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
        
    def partial_bn_explanation_transform(self, data_dict: Dict[str, List[Any]], embedding_key: str = "embeddings"):
        if embedding_key in data_dict.keys():
            if self.cluster_model is not None:
                embedding = np.array(data_dict[embedding_key], dtype="float64")
                print(f"Embedding shape {embedding.shape}")
                if embedding.ndim == 1:
                    embedding = embedding.reshape(1, -1)
                cluster = self.cluster_model.predict(embedding)
                print(f"cluster: {cluster}")
                del data_dict[embedding_key]
                data_dict["cluster"] = [cluster[0]]
        template_dict = dfv.default_sample.copy()
        keys = list(data_dict.keys())
        print(f"keys: {keys}")
        for key in template_dict.keys():
            if key in data_dict.keys():
                template_dict[key] = data_dict[key]
        # transform data  
        if self.bn_model_preprocessing is not None:
            # transform data in dataframe
            data_df = pd.DataFrame.from_dict(template_dict, orient='index').T
            # getting named transformers and transform columns
            x_out = self.bn_model_preprocessing.transform(data_df)
            feature_names = self.bn_model_preprocessing.get_feature_names_out()
            transformed_feature_names = [replace_bad_characters(feat) for feat in feature_names]
            x_out_df = pd.DataFrame(data=x_out, columns=transformed_feature_names)
            answer_dict = x_out_df.to_dict(orient='list')
            # get mapping between input and output features
            # filter original  keys vs dict 
            final_dict = {}
            for key in keys:
                for trans_name in answer_dict.keys():
                    if key.lower() in trans_name:
                        print(f"match: {trans_name}")
                        final_dict[trans_name] = answer_dict[trans_name]
            print(f"answer_dict: {final_dict}")
            return final_dict
        else:
            return data_dict
        
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
            if raw_xai.get("identity__y_pred=1", None) is not None:
                prediction_text = f"like with probability {np.round(raw_xai['identity__y_pred=1'], decimals)}"
                del raw_xai['identity__y_pred=1']
            elif raw_xai.get("identity__y_pred=0", None) is not None:
                if raw_xai['identity__y_pred=0'] < 0.5:
                    prediction_text = f"dislike with probability {np.round(1.0 - raw_xai['identity__y_pred=0'], decimals)}"
                else:
                    prediction_text = f"dislike with probability {np.round(raw_xai['identity__y_pred=0'], decimals)}"
                del raw_xai['identity__y_pred=0']
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
        evidence_dict = self.partial_bn_explanation_transform(evidence_dict, embedding_key)
        print(f"{evidence_dict.keys()} dict before")
        # prepare the evidence dictionary
        for key in evidence_dict.keys():
            if not isinstance(evidence_dict[key], List):
                evidence_dict[key] = [evidence_dict[key]]
        # predict y with the evidence and cluster
        print(f"{evidence_dict.keys()} dict before prediction")
        y_pred_partial = self.predict_with_partial_information(evidence_dict.copy())
        evidence_dict["identity__y_pred"] = [int(y_pred_partial[0]["identity__y_pred"])]
        print(f"Y partial: {y_pred_partial}")
        print(f"dict after prediction: {evidence_dict.keys()}")
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
    base_path = pth.Path(__file__).parent.parent
    print(f"Base path: {base_path}")
    model_path = os.path.join(base_path, 
                              "model_assets", 
                              "training_model_0_use_full_inputs_user_food_context_input_shape_new_x_bert_regression.tf")
    recommender_service = RecommenderService(model_path)
    model_inputs_and_type = recommender_service.get_model_inputs_and_type()
    print(model_inputs_and_type)
    precomputed_embeddings = os.path.join(base_path, 
                                          "model_assets",
                                          "full_recipe_embedding_BERT_v2_17_may_recipeId.npz")
    recommender_service.load_embeddings(precomputed_embeddings)
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
    recommended_items = recommender_service.recommend_items(sample)
    print(recommended_items)
    columns_dict = recommender_service.get_model_data_types()
    print(columns_dict)
    # Explanation
    explanation_service = ExplanationService()
    cluster_path = os.path.join(base_path,
                                "model_assets",
                                "new_experiment_complex_model_bert_cluster_full_model.pkl")
    explanation_service.load_cluster_model(cluster_path)
    path_to_ruleset = os.path.join(base_path,
                                   "model_assets",
                                   "new_experiments_ruleset_bert_0_Full_model.pkl")
    explanation_service.load_rule_set(path_to_ruleset)
    path_preprocessing_ruleset = os.path.join(base_path,
                                              "model_assets",
                                              "preprocessor_rules_new_model_bert.pkl")
    explanation_service.load_rule_set_preprocessing(path_preprocessing_ruleset)
    path_to_bn_learn = os.path.join(base_path,
                                    "model_assets",
                                    "bn_learn_model_bert_0_Full_model.pkl")
    explanation_service.load_bn_model(path_to_bn_learn)
    path_to_bn_learn_preprocessor = os.path.join(base_path,
                                                 "model_assets",
                                                 "preprocessor_bn_bert_0_Full_model.pkl")
    explanation_service.load_bn_preprocessing(path_to_bn_learn_preprocessor)
    #TODO: load preprocessing for bn and create a preprocessing function for the bn model 
    # explanation_service.load_discretized_dict('model_assets/discretize_dict.pkl')
    # transform sample 
    X_final = explanation_service.data_preprocessing_for_rules(sample, embedding_cols='embeddings')
    #X_final = np.repeat(X_final, 4, axis=0)
    print()
    print(X_final.shape)
    rule_prediction = explanation_service.explain_decision_with_rules(X_final)
    print(f"Explanation: {rule_prediction}")
    expa = explanation_service.generate_text_explanation_from_rule_prediction(rule_prediction)
    print(f"Text Explanation: {expa}")
    processed_sample = sample.copy()
    processed_sample.update({'y_pred': rule_prediction[0]})
    X_bn_learn = explanation_service.data_preprocessing_for_bn_with_pipeline(processed_sample)
    print(f"post processing: {X_bn_learn}")
    X_bn_learn_dict = X_bn_learn.to_dict(orient="list")
    print(f"dict from df: {X_bn_learn_dict}")
    expa_bn = explanation_service.explanation_bayesian_network(X_bn_learn_dict)
    print(f"Bayesian Network Explanation: {expa_bn}")
    # processed_sample = sample.copy()
    # processed_sample.update({'y_pred': rule_prediction[0]})
    # post_processed_sample = explanation_service.data_preprocessing_for_bn(processed_sample)
    # print(f"Post: {post_processed_sample}")
    # expa_bn = explanation_service.explanation_bayesian_network(post_processed_sample)
    # print(f"Bayesian Network Explanation: {expa_bn}")
    