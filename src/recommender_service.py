import tensorflow as tf
from typing import Dict, Any, List, Tuple
import traceback
import numpy as np
from model_utils import identify_data_types
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
import default_values as dfv
import string
#TODO: implement safety check on categorical inputs to the model
class RecommenderService:
    def __init__(self, model_path) -> None:
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully!")
        self.embeddings = None
        self.embedding_transformer = None
        
    def load_embedding_transformer(self, transformer_type: str = "bert") -> None:
        if transformer_type == "bert":
            self.embedding_transformer = SentenceTransformer('bert-base-nli-mean-tokens')
        
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
        # check safety for users 
        for key, value in data.items():
            if key == 'allergy':
                converted_value = []
                for item in value:
                    if item in dfv.safety_allergies:
                        converted_value.append(item)
                    else:
                        converted_value.append(dfv.safety_allergies[0])
                value = converted_value
            if key == 'meal_type_y':
                converted_value = []
                for item in value:
                    if item in dfv.meal_type_y:
                        converted_value.append(item)
                    else:
                        converted_value.append('NotInformation')
                value = converted_value
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
                # safe allergy user 
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
            
            
    def check_compatibility(self, user_profile, context, recipe_data):
        #TODO: Impute missing values for new recipes with know recipes
        #TODO: Control when is not a transformer for this  
        # Transform ingredients 
        ingredients = recipe_data["ingredients"]
        new_recipe_data = {}
        final_dict = {}
        new_recipe_data.update(recipe_data)
        if self.embedding_transformer is not None:
            del new_recipe_data["ingredients"]
            embedding = self.embedding_transformer.encode(ingredients)
            print(f"Embedding shape: {embedding.shape}")
            if embedding.ndim == 1:
                new_recipe_data["embeddings"] = embedding.reshape(1, -1)
            else:
                new_recipe_data["embeddings"] = embedding
        print(f"shape embedding: {new_recipe_data['embeddings'].shape}")
        # 1. Make the prediction 
        final_dict.update(user_profile)
        final_dict.update(context)
        final_dict.update(new_recipe_data)
        for key, value in final_dict.items():
            if not isinstance(value, List) and not isinstance(value, np.ndarray):
                final_dict[key] = [value]
        input_data = self.transform_input_data(final_dict)
        print(f"Input data: {input_data}")
        predictions = self.model(input_data)
        predictions_numpy = predictions.numpy()
        print("Predictions:", predictions_numpy)
        return final_dict, predictions_numpy
            
    def get_model_data_types(self) -> Dict[str, Any]:
        dict_ans = {}
        if self.model is not None:
            model_inputs = self.model.inputs
            numeric_feat, categorical_feat, embed_feat = identify_data_types(model_inputs)
            dict_ans['numeric_features'] = numeric_feat
            dict_ans['categorical_features'] = categorical_feat
            dict_ans['embedding_features'] = embed_feat
        return dict_ans
            
    def recommend_by_ingredients_similarity(self, 
                                            user_profile: Dict, 
                                            context: Dict, 
                                            recipe_ingredients: str, 
                                            num_items: int = 2,
                                            sample_size: int = 500):
        recipes_df = pd.read_csv("model_assets/df_recipes.csv", sep='|', index_col=0)
        # process recipe ingredients 
        text = recipe_ingredients.lower()
        translator = str.maketrans('', '', string.punctuation)
        # Use the translate method to remove the punctuation
        cleaned_text = text.translate(translator)
        splitted_text = cleaned_text.split(' ')
        similar_recipes = []
        if len(splitted_text) > 0:
            # check in the database 
            if "ingredients_list" in recipes_df.columns:
                for ingredient in splitted_text:
                    mask = recipes_df['ingredients_list'].str.contains(ingredient)
                    recipes = recipes_df.loc[mask, "recipeId"].to_list()
                    similar_recipes.extend(recipes)
        similar_recipes_ids = list(set(similar_recipes))      
        # 1. extract embedding 
        if self.model is not None and self.embedding_transformer is not None:
            embedding = self.embedding_transformer.encode(recipe_ingredients)
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
        # 2. pick up most similar recipes 
        recipe_list = []
        recipe_ids = []
        for recipe_id in self.embeddings.keys():
            recipe_ids.append(recipe_id)
            recipe_list.append(self.embeddings[recipe_id])
        recipe_matrix = np.array(recipe_list)
        print(f"Recipes matrix shape: {recipe_matrix.shape}")
        distances = cosine_distances(recipe_matrix, embedding)
        distances_dict = dict(zip(recipe_ids, distances))
        sorted_recipes = list(sorted(distances_dict.items(), key=lambda item: item[1], reverse=True))
        chose_recipes = [item[0] for item in sorted_recipes]
        chose_recipes = chose_recipes[:sample_size]
        chosen_set = list(set(chose_recipes).union(set(similar_recipes)))
        print(f"Chosen set len: {len(chosen_set)}")
        print(f"Chosen recipes:{chose_recipes[:sample_size]}")
        recipes_samples = recipes_df[recipes_df["recipeId"].isin(chosen_set)]
        if recipes_samples.shape[0] > sample_size:
            recipes_samples = recipes_samples.sample(sample_size)
        # 3. execute recommendation 
        topk_recommendations, topk_indices, final_dict, top_pred = self.produce_recommendations(user_data=user_profile,
                                                            context_data=context,
                                                            recipes_df=recipes_samples,
                                                            num_items=num_items,
                                                            )
        # 4. return the hights ranked recipes 
        return topk_recommendations, topk_indices, final_dict, top_pred
        