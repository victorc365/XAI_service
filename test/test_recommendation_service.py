import sys 
import os 
import pytest
from pathlib import Path
import pandas as pd 
import numpy as np 

# add modules to system path 
current_file_parent = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.path.join(current_file_parent, 'src'))
# import testing modules 
from recommender_service import RecommenderService

@pytest.fixture
def recommender_service():
    path_to_model = os.path.join(current_file_parent, 'model_assets', 'training_model_0_use_full_inputs_user_food_context_input_shape_new_x_bert_regression.tf')
    recommender_service = RecommenderService(path_to_model)
    recommender_service.load_embedding_transformer()
    embedding_path = os.path.join(current_file_parent, 'model_assets', 'full_recipe_embedding_BERT_v2_17_may_recipeId.npz')
    recommender_service.load_embeddings(embedding_path)
    return recommender_service

@pytest.fixture
def get_test_data_for_model():
    test_data_for_model =     sample = {'BMI': ['healthy'], 
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
     'recipe_id': "food_0", 
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
    return test_data_for_model

@pytest.fixture
def load_query_data():
    query_data = {
    "profile": {
        "nutrition_goal": "maintain_fit",
        "clinical_gender": "F",
        "age_range": "30-39",
        "life_style": "Very active",
        "weight": 65,
        "height": 170,
        "projected_daily_calories": 2200,
        "current_daily_calories": 1700,
        "cultural_factor": "vegan_observant",
        "allergy": "soy",
        "current_working_status": "Unemployed",
        "marital_status": "Single",
        "ethnicity": "White",
        "BMI": "healthy",
        "next_BMI": "healthy"
        },
    "context": {
        "day_number": 1,
        "meal_type_x": "lunch",
        "time_of_meal_consumption": 12.01,
        "place_of_meal_consumption": "home",
        "social_situation_of_meal_consumption": "alone"
        }
    }
    return query_data

@pytest.fixture
def data_for_compatibility_checking():
    query_data = {
    "profile": {
        "nutrition_goal": "maintain_fit",
        "clinical_gender": "F",
        "age_range": "30-39",
        "life_style": "Very active",
        "weight": 65,
        "height": 170,
        "projected_daily_calories": 2200,
        "current_daily_calories": 1700,
        "cultural_factor": "vegetarian_observant",
        "allergy": "soy",
        "current_working_status": "Unemployed",
        "marital_status": "Single",
        "ethnicity": "White",
        "BMI": "healthy",
        "next_BMI": "healthy"
    },
    "context": {
        "day_number": 1,
        "meal_type_x": "lunch",
        "time_of_meal_consumption": 12.01,
        "place_of_meal_consumption": "home",
        "social_situation_of_meal_consumption": "alone"
        },
    "recipe_data": {
     "allergens": "soy", 
     "calories": 650, 
     "carbohydrates": 39,
     "cultural_restriction": "halal",  
     "ingredients": "Tomatoes, cheese, bread",
     "fat": 30, 
     "fiber": 29,
     "price": 2, 
     "protein": 30, 
     "taste": "sweet"
        }
    }
    return query_data

@pytest.fixture
def data_recommendation_ingredient_similarity():
    query_data = {
    "profile": {
        "nutrition_goal": "maintain_fit",
        "clinical_gender": "F",
        "age_range": "30-39",
        "life_style": "Very active",
        "weight": 65,
        "height": 170,
        "projected_daily_calories": 2200,
        "current_daily_calories": 1700,
        "cultural_factor": "vegan_observant",
        "allergy": "soy",
        "current_working_status": "Unemployed",
        "marital_status": "Single",
        "ethnicity": "White",
        "BMI": "healthy",
        "next_BMI": "healthy"
       },
        "context": {
        "day_number": 1,
        "meal_type_x": "lunch",
        "time_of_meal_consumption": 12.01,
        "place_of_meal_consumption": "home",
        "social_situation_of_meal_consumption": "alone"
        },
        "ingredients": "tomatoes, cheese"
    }
    return query_data

@pytest.fixture
def load_recipes_data():
    path_to_recipes = os.path.join(current_file_parent,"model_assets", "df_recipes.csv")
    recipes_df = pd.read_csv(path_to_recipes, sep="|", index_col=0)
    print(f"Loaded recipes: {recipes_df.shape}")
    return recipes_df
    

def test_get_model_inputs_and_type(recommender_service):
    model_inputs = recommender_service.get_model_inputs_and_type()
    assert model_inputs is not None
    
def test_get_embedding_for_recipe_id(recommender_service):
    recipe_id = 'food_1250'
    embedding = recommender_service.get_embedding_for_recipe_id(recipe_id)
    assert embedding is not None
    
def test_transform_input_data(recommender_service, get_test_data_for_model):
    transformed_data = recommender_service.transform_input_data(get_test_data_for_model)
    assert transformed_data is not None and len(transformed_data) > 0
    
def test_get_model_inputs_and_type(recommender_service):
    model_inputs = recommender_service.get_model_inputs_and_type()
    assert model_inputs is not None
    
def test_recommend_items(recommender_service, get_test_data_for_model):
    # get embedding 
    data = get_test_data_for_model.copy()
    recipe_id = get_test_data_for_model['recipe_id']
    embedding = recommender_service.get_embedding_for_recipe_id(recipe_id)
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)
    data['embeddings'] = embedding
    top_recommendations, top_indices = recommender_service.recommend_items(data)
    assert top_recommendations is not None and top_indices is not None
    
def test_produce_recommendations(recommender_service, load_query_data, load_recipes_data):
    query_data = load_query_data
    recipes = load_recipes_data
    user_profile = query_data['profile']
    context_data = query_data['context']
    topk_recommendations, topk_indices, final_dict, top_pred = recommender_service.produce_recommendations(
                                user_data = user_profile, 
                                context_data = context_data,  
                                recipes_df = recipes.sample(1000),
                                num_items=5)
    assert len(topk_recommendations) == 5
    
def test_get_model_data_types(recommender_service):
    input_data_types = recommender_service.get_model_data_types()
    assert input_data_types is not None and len(input_data_types) > 0
    

def test_check_compatibility(recommender_service, data_for_compatibility_checking):
    data = data_for_compatibility_checking.copy()
    user_profile = data['profile']
    context = data['context']
    recipe_data = data["recipe_data"]
    final_dict, prediction_numpy = recommender_service.check_compatibility(user_profile, context, recipe_data)
    assert final_dict is not None and prediction_numpy is not None
    
def test_recommend_by_ingredients_similarity(recommender_service, data_recommendation_ingredient_similarity):
    data = data_recommendation_ingredient_similarity.copy()
    user_profile = data['profile']
    context = data['context']
    recipe_ingredients = data['ingredients']
    topk_recommendations, topk_indices, final_dict, top_pred = recommender_service.recommend_by_ingredients_similarity(
                                            user_profile=user_profile, 
                                            context=context, 
                                            recipe_ingredients=recipe_ingredients)
    assert topk_recommendations is not None
    
    
