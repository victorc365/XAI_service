import tensorflow as tf
from typing import Dict, Any, List, Tuple
import traceback
import numpy as np
from model_utils import identify_data_types
import pandas as pd 

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
        else:
            return np.nan
        
    def transform_input_data(self, data: Dict[str, str]) -> Dict[str, Any]:
        data_dict = {}
        input_data_types = self.get_model_inputs_and_type()
        for key, value in data.items():
            if key in input_data_types.keys():
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
        
    def recommend_items(self, data: Dict, topk: int = 2) -> Tuple[List[float], int]:
        try:
            if self.model is not None:
                input_data = self.transform_input_data(data)
                predictions = self.model(input_data)
                predictions_numpy = predictions.numpy()
                print("Predictions:", predictions_numpy)
                topk_indices = predictions_numpy.argsort()[-topk:][::-1]  # get topk indices in descending
                topk_recommendations = predictions_numpy[topk_indices]
                return topk_recommendations, topk_indices
        except Exception as e:
            print(f"An error occurred while predicting items: {traceback.format_exc()}")
            raise Exception(f"An error occurred while predicting items: {traceback.format_exc()}")
        
    def produce_recommendations(self, 
                                user_data: Dict, 
                                context_data: Dict, 
                                recipes_df: pd.DataFrame,
                                num_items: int = 2, 
                                recipe_id_col: str = 'recipeId'):
        try:
            if self.model is not None:
                final_dict = {}
                # generate_embeddings
                recipes_df['embeddings'] = recipes_df[recipe_id_col].apply(lambda x: self.get_embedding_for_recipe_id(x))
                mask = recipes_df['embeddings'].isna()
                recipes_df_final = recipes_df.loc[~mask, :]
                num_recipes = recipes_df_final.shape[0]
                print(f"Num  candidates: {num_recipes}")
                final_dict.update(user_data)
                final_dict.update(context_data)
                print(f"Final dict: {final_dict}")
                for key, value in final_dict.items():
                    if not isinstance(value, List):
                        final_dict[key] = [value] * num_recipes
                recipes_dict = recipes_df_final.to_dict(orient='list')
                final_dict.update(recipes_dict)
                input_data = self.transform_input_data(final_dict)
                print(f"Input data: {input_data}")
                predictions = self.model(input_data)
                predictions_numpy = predictions.numpy()
                print("Predictions:", predictions_numpy)
                topk_indices = np.argsort(predictions_numpy.flatten())[::-1][:num_items]  # get topk indices in descending
                topk_indices = topk_indices.tolist()
                print("Topk indices:", topk_indices)
                recipes_ids = recipes_df[recipe_id_col].tolist()
                topk_recommendations = [recipes_ids[i] for i in topk_indices]
                top_predictions = [predictions_numpy.flatten()[i] for i in topk_indices]
                return topk_recommendations, topk_indices, final_dict, top_predictions
        except Exception as e:
            print(f"Error generating recipes: {e}")
            print(traceback.format_exc())
        
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
    # load recipes dataframe 
    recipes_df = pd.read_csv('model_assets/df_recipes.csv', sep='|', index_col=0)
    print(f'Recipes loaded: {recipes_df.shape}')
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
    # test the second function example 2 
    user_data = {'nutrition_goal': 'maintain_fit', 
                 'clinical_gender': 'M', 
                 'age_range': '30-39',
                 'life_style': 'Sedentary', 
                 'weight': 65, 
                 'height': 165,
                 'projected_daily_calories': 2200,
                 'current_daily_calories': 1700,
                 'cultural_factor': 'vegan_observant', 
                 'allergy': 'soy',
                'current_working_status': 'Unemployed', 
                'marital_status': 'Single',
                'ethnicity': 'White',
                'BMI': 'healthy', 
                'next_BMI': 'healthy'
                }
    context_data = {'day_number': 1,
                    'meal_type_y': 'lunch',
                    'time_of_meal_consumption': 12.01, 
                    'place_of_meal_consumption': 'home', 
                    'social_situation_of_meal_consumption': 'alone'}
    recommendations, indices, final_dict = recommender_service.produce_recommendations(user_data, context_data, recipes_df.sample(n=10))
    print(f"recommendations recipes: {recommendations}")
    