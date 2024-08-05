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

# Init objects
recommender_service = RecommenderService("model_assets/model_full_use__fold_0.tf")
model_inputs_and_type = recommender_service.get_model_inputs_and_type()
print(model_inputs_and_type)
recommender_service.load_embeddings("model_assets/full_recipe_embedding_BERT_v2_17_may_recipeId.npz")
explanation_service = ExplanationService()
explanation_service.load_cluster_model('model_assets/new_experiment_bert_cluster_full_model.pkl')
explanation_service.load_rule_set('model_assets/new_experiments_ruleset_bert_0.pkl')
explanation_service.load_bn_model('model_assets/bn_model_bert.pkl')
explanation_service.load_rule_set_preprocessing('model_assets/preprocessor_rule.pkl')
explanation_service.load_discretized_dict('model_assets/discretize_dict.pkl')



class Recommendation(BaseModel):
    recipe_id: str
    probability: float

app = FastAPI(
    title="Recommendation and explanation model",
    version=1.0,
    description="API for recommendation and explanation model"
)


@app.get("/")
def root():
    return {"message": "Welcome to the recommendation and explanation model API"}

@app.get("/healthcheck/")
def healthcheck():
    return {"status": "OK"}

@app.post("/recommendation/")
async def recommendation(data: Request, num_recommendations: int = 2):
    try:
        input_data = await data.json()
        print(f"input data: {input_data}")
        user_dict = input_data["profile"]
        context_dict = input_data["context"]
        recipes_df = pd.read_csv("model_assets/df_recipes.csv", sep='|', index_col=0)
        topk_recommendations, topk_indices, final_dict = recommender_service.produce_recommendations(user_data=user_dict,
                                                            context_data=context_dict,
                                                            recipes_df=recipes_df.sample(100),
                                                            num_items=num_recommendations
                                                            )
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
        post_processed_sample = explanation_service.data_preprocessing_for_bn(processed_sample)
        print(f"Post: {post_processed_sample}")
        expa_bn = explanation_service.generate_text_explanation_from_bn_prediction(post_processed_sample)
        print(f"Bayesian Network Explanation: {expa_bn}")
        answer = {
            "recommendations": topk_recommendations,
            "rule_based_explanation": expa,
            "probabilistic_explanation": expa_bn
        }
        answer_encode = jsonable_encoder(answer)
        return JSONResponse(content=answer_encode)
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"An error occurred while generating items: {str(e)}")




@app.post("/recommend/")
async def predict_items(data: Request, num_recommendations: int = 2):
    try:
        input_data = await data.json()
        print(f"Input data: {input_data}")
        new_data = input_data.copy()
        del new_data["recipeId"]
        new_data["embeddings"] = recommender_service.get_embedding_for_recipe_id(input_data["recipeId"]).reshape(1, -1)
        recommendation_results, reco_index = recommender_service.recommend_items(new_data, num_recommendations)
        X_final = explanation_service.data_preprocessing_for_rules(new_data, embedding_cols='embeddings')
        #X_final = np.repeat(X_final, 4, axis=0)
        print()
        print(X_final.shape)
        rule_prediction = explanation_service.explain_decision_with_rules(X_final)
        expa = explanation_service.generate_text_explanation_from_rule_prediction(rule_prediction)
        print(f"Text Explanation: {expa}")
        processed_sample = new_data.copy()
        processed_sample.update({'y_pred': rule_prediction[0]})
        post_processed_sample = explanation_service.data_preprocessing_for_bn(processed_sample)
        print(f"Post: {post_processed_sample}")
        expa_bn = explanation_service.explanation_bayesian_network(post_processed_sample)
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