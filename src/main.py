import traceback
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn
from recommender_service import RecommenderService
from explanation_service import ExplanationService
from typing import List
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import pandas as pd
from sentence_transformers import SentenceTransformer
import pathlib as pth
import os
import numpy as np

# loading 
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
recommender_service.load_embedding_transformer()
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
path_to_bn_learn_model = os.path.join(base_path,
                                "model_assets",
                                "bn_learn_model_bert_0_Full_model.pkl")
explanation_service.load_bn_model(path_to_bn_learn_model)
path_to_bn_learn_preprocessor = os.path.join(base_path,
                                                "model_assets",
                                                "preprocessor_bn_bert_0_Full_model.pkl")
explanation_service.load_bn_preprocessing(path_to_bn_learn_preprocessor)

model_bert_st = SentenceTransformer('bert-base-nli-mean-tokens')


class Recommendation(BaseModel):
    recipe_id: str
    probability: float
    
def multi_sampling(recipes_df, user_dict, context_dict, num_recommendations=2, sample_size=100, threshold=0.7):
    while True:
        sample = recipes_df.sample(n=sample_size, replace=False)
        topk_recommendations, topk_indices, final_dict, top_pred = recommender_service.produce_recommendations(
                user_data=user_dict,
                context_data=context_dict,
                recipes_df=sample,
                num_items=num_recommendations
                                            )
        if all(pred >= threshold for pred in top_pred):
            return topk_recommendations, topk_indices, final_dict, top_pred
        
        

app = FastAPI(
    title="Recommendation and explanation model",
    version=1.0,
    description="API for recommendation and explanation model"
)


@app.get("/")
def root():
    return {"message": "Welcome to the recommendation and explanation model API"}

@app.get("/healthcheck/")
def healthCheck():
    return {"status": "OK"}

@app.post("/recommendation/")
async def recommendation(data: Request, num_recommendations: int = 2, sample_size: int = 100):
    #TODO: completed ready to check only to check security
    try:
        input_data = await data.json()
        print(f"input data: {input_data}")
        user_dict = input_data["profile"]
        context_dict = input_data["context"]
        recipes_df = pd.read_csv("model_assets/df_recipes.csv", sep='|', index_col=0)
        if "recipes" in input_data.keys():
            recipes_ids = input_data["recipes"]
            recipes_samples = recipes_df[recipes_df["recipeId"].isin(recipes_ids)]
            topk_recommendations, topk_indices, final_dict, top_pred = recommender_service.produce_recommendations(user_data=user_dict,
                                                            context_data=context_dict,
                                                            recipes_df=recipes_samples,
                                                            num_items=num_recommendations
                                                            )
        else:
            topk_recommendations, topk_indices, final_dict, top_pred = multi_sampling(recipes_df, 
                                                                                      user_dict, 
                                                                                      context_dict, 
                                                                                      num_recommendations=num_recommendations, 
                                                                                      sample_size=sample_size, 
                                                                                      threshold=0.7)
        recipe_id_mask = recipes_df["recipeId"].isin(topk_recommendations)
        recipe_names = recipes_df.loc[recipe_id_mask, "name"].to_list()
        # check high values in recommendations mode
        general_explanation = explanation_service.generate_high_level_explanation(entities_list=["profile", "recipe", "context"],
                                                                                  predictions=top_pred)
        ans = {}
        for key in final_dict.keys():
            ans[key] = [final_dict[key][i] for i in topk_indices]
        print(f"reco_dict: {ans}")
        X_final = explanation_service.data_preprocessing_for_rules(ans, embedding_cols='embeddings')
        print(X_final.shape)
        rule_prediction = explanation_service.explain_decision_with_rules(X_final)
        rule_explanation = explanation_service.generate_text_explanation_from_rule_prediction(rule_prediction)
        print(f"Text Explanation: {rule_explanation}")
        processed_sample = ans.copy()
        processed_sample.update({'y_pred': rule_prediction[0]})
        X_bn_learn = explanation_service.data_preprocessing_for_bn_with_pipeline(processed_sample)
        print(f"post processing: {X_bn_learn}")
        X_bn_learn_dict = X_bn_learn.to_dict(orient="list")
        print(f"dict from df: {X_bn_learn_dict}")
        expa_bn = explanation_service.generate_text_explanation_from_bn_prediction(X_bn_learn_dict)
        print(f"Bayesian Network Explanation: {expa_bn}")
        answer = {
            "recipe_names": recipe_names,
            "recommendations": topk_recommendations,
            "general_explanation": general_explanation,
            "rule_based_explanation": rule_explanation,
            "probabilistic_explanation": expa_bn
        }
        answer_encode = jsonable_encoder(answer)
        return JSONResponse(content=answer_encode)
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"An error occurred while generating items: {str(e)}")
    
@app.post("/checkCompatibility/")
async def check_new_recipe(data: Request):
    try:
        input_data = await data.json()
        user_profile = input_data["profile"]
        context = input_data["context"]
        recipe_data = input_data["recipe_data"]
        final_dict, predictions = recommender_service.check_compatibility(user_profile, context, recipe_data)
        general_explanation = explanation_service.generate_high_level_explanation(entities_list=["profile", "recipe", "context"],
                                                                                  predictions=predictions)
        print(f"General explanation: {general_explanation}")
        X_final = explanation_service.data_preprocessing_for_rules(final_dict, embedding_cols='embeddings')
        print(f"X_final: {X_final.shape}")
        rule_prediction = explanation_service.explain_decision_with_rules(X_final)
        general_explanation = explanation_service.generate_high_level_explanation(entities_list=["profile", "recipe", "context"],
                                                                                  predictions=rule_prediction[0].tolist())
        print(f"Rule prediction: {rule_prediction}")
        expa = explanation_service.generate_text_explanation_from_rule_prediction(rule_prediction)
        print(f"Text Explanation: {expa}")
        processed_sample = final_dict.copy()
        processed_sample.update({'y_pred': rule_prediction[0]})
        X_bn_learn = explanation_service.data_preprocessing_for_bn_with_pipeline(processed_sample)
        print(f"post processing: {X_bn_learn}")
        X_bn_learn_dict = X_bn_learn.to_dict(orient="list")
        print(f"dict from df: {X_bn_learn_dict}")
        expa_bn = explanation_service.generate_text_explanation_from_bn_prediction(X_bn_learn_dict)
        print(f"Bayesian Network Explanation: {expa_bn}")
        answer = {
            "recommendations": [],
            "general_explanation": general_explanation,
            "rule_based_explanation": expa,
            "probabilistic_explanation": expa_bn
        }
        answer_encode = jsonable_encoder(answer)
        return JSONResponse(content=answer_encode)
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"An error occurred while processing new recipe: {str(e)}")

@app.post("/recommendByProximity/")
async def recommend_by_proximity(data: Request, num_recommendations: int = 2): 
    # Recommend by proximity given user query
    try:
        recipes_df = pd.read_csv("model_assets/df_recipes.csv", sep='|', index_col=0)
        input_data  = await data.json()
        print(f"Input data: {input_data}")
        profile = input_data['profile']
        context = input_data['context']
        ingredients = input_data['ingredients']
        topk_recommendations, topk_indices, final_dict, top_pred = recommender_service.recommend_by_ingredients_similarity(
                                                                user_profile=profile,
                                                                context=context,
                                                                recipe_ingredients=ingredients,
                                                                num_items=num_recommendations)
        recipe_id_mask = recipes_df["recipeId"].isin(topk_recommendations)
        recipe_names = recipes_df.loc[recipe_id_mask, "name"].to_list()
        # check high values in recommendations mode
        general_explanation = explanation_service.generate_high_level_explanation(entities_list=["profile", "recipe", "context"],
                                                                                 predictions=top_pred)
        ans = {}
        for key in final_dict.keys():
            ans[key] = [final_dict[key][i] for i in topk_indices]
        print(f"reco_dict: {ans}")
        X_final = explanation_service.data_preprocessing_for_rules(ans, embedding_cols='embeddings')
        print(X_final.shape)
        rule_prediction = explanation_service.explain_decision_with_rules(X_final)
        expa = explanation_service.generate_text_explanation_from_rule_prediction(rule_prediction)
        print(f"Text Explanation: {expa}")
        processed_sample = ans.copy()
        processed_sample.update({'y_pred': rule_prediction[0]})
        X_bn_learn = explanation_service.data_preprocessing_for_bn_with_pipeline(processed_sample)
        print(f"post processing: {X_bn_learn}")
        X_bn_learn_dict = X_bn_learn.to_dict(orient="list")
        print(f"dict from df: {X_bn_learn_dict}")
        expa_bn = explanation_service.generate_text_explanation_from_bn_prediction(X_bn_learn_dict)
        print(f"Bayesian Network Explanation: {expa_bn}")
        answer = {
            "recipe_names": recipe_names,
            "recommendations": topk_recommendations,
            "general_explanation": general_explanation,
            "rule_based_explanation": expa,
            "probabilistic_explanation": expa_bn
        }
        answer_encode = jsonable_encoder(answer)
        return JSONResponse(content=answer_encode)
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"An error occurred while processing new recipe: {str(e)}")
    
@app.post("/partialQuery/")
async def partialQuery(data: Request):
    try:
        input_data = await data.json()
        user_profile = input_data["profile"]
        context = input_data["context"]
        partial_recipe = input_data["recipe"]
        # preprocess and prepare for query the probabilistic model
        if "ingredients" in partial_recipe:
            # transform into embedding
            embedding = model_bert_st.encode(partial_recipe["ingredients"])
            print(f"Embedding: {embedding}")
            if embedding.ndim == 1:
                partial_recipe["embeddings"] = embedding.reshape(1, -1)
            else:
                partial_recipe["embeddings"] = embedding
        processed_sample = {}
        processed_sample.update(user_profile)
        processed_sample.update(context)
        processed_sample.update(partial_recipe)
        for key in processed_sample.keys():
            if not isinstance(processed_sample[key], (list, np.ndarray)):
                processed_sample[key] = [processed_sample[key]]
        #TODO: Adapt current code for partial query with the new method 
        partial_explanation = explanation_service.generate_partial_explanation_and_predictions(processed_sample)
        answer = {
            "partial_probabilistic_explanation": partial_explanation
        }
        answer_encoded = jsonable_encoder(answer)
        return JSONResponse(content=answer_encoded)
    except Exception as e:
        print(f"Error occurred while generating recipes: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"An error occurred while generating items: {str(e)}")
    

@app.post("/recommend/")
async def predict_items(data: Request, num_recommendations: int = 2):
    #TODO: restructure or remove it.
    try:
        input_data = await data.json()
        print(f"Input data: {input_data}")
        new_data = input_data.copy()
        ingredients = new_data["ingredients"]
        #TODO: preprocess ingredients list
        if len(ingredients) == 1:
            embedding = model_bert_st.encode(ingredients)
            if embedding.ndim == 1:
                new_data["embeddings"] = embedding.reshape(1, -1)
            else:
                new_data["embeddings"] = embedding
        elif len(ingredients) > 1:
            embedding = model_bert_st.encode(ingredients)
            new_data["embeddings"] = embedding.tolist()
        recommendation_results, reco_index = recommender_service.recommend_items(new_data, num_recommendations)
        print("Prediction successful")
        X_final = explanation_service.data_preprocessing_for_rules(new_data, embedding_cols='embeddings')
        #X_final = np.repeat(X_final, 4, axis=0)
        print(f"X_final obtained shape: {X_final.shape}")
        print(X_final.shape)
        rule_prediction = explanation_service.explain_decision_with_rules(X_final)
        expa = explanation_service.generate_text_explanation_from_rule_prediction(rule_prediction)
        print(f"Text Explanation: {expa}")
        processed_sample = new_data.copy()
        processed_sample.update({'y_pred': rule_prediction[0]})
        X_bn_learn = explanation_service.data_preprocessing_for_bn_with_pipeline(processed_sample)
        print(f"post processing: {X_bn_learn}")
        X_bn_learn_dict = X_bn_learn.to_dict(orient="list")
        print(f"dict from df: {X_bn_learn_dict}")
        expa_bn = explanation_service.generate_text_explanation_from_bn_prediction(X_bn_learn_dict)
        print(f"Bayesian Network Explanation: {expa_bn}")
        answer  = {"reco_idx": reco_index.tolist(),
                   "rule_explanation": expa,
                   "bn_explanation": expa_bn}
        answer_encoded = jsonable_encoder(answer)
        return JSONResponse(content=answer_encoded)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while predicting items: {str(e)}")



if __name__ == "__main__":
    uvicorn.run("main:app", 
                host="localhost", 
                port=8500,
                reload=True)