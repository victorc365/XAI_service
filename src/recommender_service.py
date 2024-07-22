import tensorflow as tf
from typing import Dict, Any, List, Tuple
import traceback
import numpy as np
from model_utils import identify_data_types

class RecommenderService:
    def __init__(self, model_path) -> None:
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully!")
        self.embeddings = None
        
    def load_embeddings(self, embeddings_path: str) -> None:
        # load embedding if it is necessary
        self.embeddings = dict(np.load(embeddings_path))
        print(f"Embedding loaded shape: {len(self.embeddings)}")
        
    def get_embedding_for_recipe_id(self, recipe_id: str) -> np.ndarray:
        if self.embeddings is not None and recipe_id in self.embeddings:
            return self.embeddings[recipe_id]
        
    def transform_input_data(self, data: Dict[str, str]) -> Dict[str, Any]:
        data_dict = {}
        input_data_types = self.get_model_inputs_and_type()
        for key, value in data.items():
            data_dict[key] = tf.convert_to_tensor(value, dtype=input_data_types[key], name=key)
        return data_dict 
        
    def get_model_inputs_and_type(self) -> Dict[str, tf.DType]:
        if self.model is not None:
            inputs = self.model.inputs
            input_types = [input.dtype for input in inputs]
            input_names = [input.name for input in inputs]
            return dict(zip(input_names, input_types))
        else:
            raise Exception("Model is not loaded.")
        
    def recommend_items(self, data: Dict, topk: int = 2) -> List[Tuple[str, float]]:
        try:
            if self.model is not None:
                input_data = self.transform_input_data(data)
                predictions = self.model(input_data)
                predictions_numpy = predictions.numpy()
                print("Predictions:", predictions_numpy)
                topk_indices = predictions_numpy.argsort()[-topk:][::-1]  # get topk indices in descending
                topk_recommendations = predictions_numpy[topk_indices]
                return topk_recommendations
        except Exception as e:
            print(f"An error occurred while predicting items: {traceback.format_exc()}")
            raise Exception(f"An error occurred while predicting items: {traceback.format_exc()}")
        
    def get_model_data_types(self) -> Dict[str, Any]:
        dict_ans = {}
        if self.model is not None:
            model_inputs = self.model.inputs
            numeric_feat, categorical_feat, embed_feat = identify_data_types(model_inputs)
            dict_ans['numeric_features'] = numeric_feat
            dict_ans['categorical_features'] = categorical_feat
            dict_ans['embedding_features'] = embed_feat
        return dict_ans
            
        
        
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