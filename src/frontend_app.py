import streamlit as st
import requests
from urllib.parse import urljoin
import datetime as dt
from typing import Union
import default_values as dfv


st.title('Test recommendation model')


st.markdown("""To test the recommender model first select the mode in the dropdown bellow and provide the data in the form
            """)
options_reco = ['Best recommendations', 'Evaluate recipe list by id', 'Find me a recipe given ingredients', 'partial query',  'Provide full data']

reco_mode = st.selectbox('Please select a recommendation mode:',
                         options=options_reco)

# Define the FastAPI endpoint
base_url = 'http://localhost:8500/'

# collect data 
st.subheader("User data", divider=True)
st.write('Please complete the user profile:')
profile = {
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
    }

profile['nutrition_goal'] = st.selectbox(f"Please select your nutrition_goal:",
                                         options=list(dfv.nutritional_goals))
profile['clinical_gender'] = st.selectbox(f"Please select your gender:",
                                         options=["M", "F"])
profile['age_range'] = st.selectbox(f"Please select your age range:",
                                         options=list(dfv.age_dict.keys()))
profile['life_style'] = st.selectbox(f"Please select your life_style:",
                                         options=dfv.user_entity['life_style'])
profile["weight"] = st.number_input(f"Enter your weight:",
                                    min_value=1.0, max_value=600.0, value=50.0)

profile["height"] = st.number_input(f"Enter your height:",
                                    min_value=50.0, max_value=250.0, value=170.0)

profile["projected_daily_calories"] = st.number_input(f"Enter your projected daily calories:",
                                                      min_value=1000, max_value=5000, value=2200)

profile["current_daily_calories"] = st.number_input(f"Enter your current daily calories:",
                                                      min_value=1000, max_value=5000, value=1700)

profile["cultural_factor"] = st.selectbox(f"Please select your cultural factor:",
                                         options=list(dfv.cultural_factors.keys()))
profile["allergy"] = st.selectbox(f"Please select your allergy:",
                                   options=dfv.user_allergies)
profile["current_working_status"] = st.selectbox(f"Please select your current working status:",
                                   options=dfv.user_entity["current_working_status"])
profile["marital_status"] = st.selectbox(f"Please select your marital status:",
                                   options=dfv.user_entity["marital_status"])
profile["ethnicity"] = st.selectbox(f"Please select your ethnicity:",
                                   options=dfv.user_entity["ethnicity"])
profile["BMI"] = st.selectbox(f"Please select your BMI:",
                                   options=list(dfv.bmi_set))
profile["next_BMI"] = st.selectbox(f"Please select your next BMI:",
                                   options=list(dfv.bmi_set))

st.subheader("Context data", divider=True)
st.write("Please complete the context:")
context = {
        "day_number": 1,
        "meal_type_x": "lunch",
        "time_of_meal_consumption": 12.01,
        "place_of_meal_consumption": "home",
        "social_situation_of_meal_consumption": "alone"
    }

def process_time(time_object: Union[dt.time, None]) -> float:
    if time_object is None:
        return 0.0
    return float(f"{time_object.hour}.{time_object.minute}")

context["day_number"] = st.number_input("Day number:", min_value=0, max_value=10000)
context["meal_type_x"] = st.selectbox("Meal type:", 
                                      options=list(dfv.meals_calorie_dict.keys()))
context["time_of_meal_consumption"] = process_time(st.time_input("Time of meal consumption: ", value="now"))
context["place_of_meal_consumption"] = st.selectbox("Place of meal consumption", 
                                                    options=list(dfv.place_proba_dict.keys()))
context["social_situation_of_meal_consumption"] = st.selectbox("Social situation of meal consumption:", 
                                                                options=list(dfv.social_situation_proba_dict.keys()))


recipe_data = {'cultural_restriction': "vegan", 
               'calories': 650, 
               'allergens': "soy", 
               'taste': "sweet", 
               'price': 2,  
               'fiber': 29, 
               'fat': 30, 
               'protein': 30,  
               'carbohydrates': 39, 
               'ingredient_list': []}
# prepare data 
data = {}
target_url = base_url
if reco_mode == options_reco[0]:
    target_url = urljoin(base_url, '/recommendation/')
    st.write("Target url: %s" % target_url)
    data['profile'] = profile
    data['context'] = context
elif reco_mode == options_reco[1]:
    target_url = urljoin(base_url, '/recommendation/')
    st.write("Target url: %s" % target_url)
    recipes_list = st.multiselect('Please choose recipes id from list:',
                                  [f'food_{i}' for i in range(0, 7017)])
    data['profile'] = profile
    data['context'] = context
    if len(recipes_list) > 0:
        data['recipes'] = recipes_list
elif reco_mode == options_reco[2]:
    target_url = urljoin(base_url, '/recommendByProximity/')
    st.subheader('Ingredients query', divider=True)
    ingredients_text = st.text_input(label='Please introduce the ingredients:')
    data['profile'] = profile
    data['context'] = context
    data["ingredients"] = ingredients_text
    st.write("Target url: %s" % target_url)
elif reco_mode == options_reco[3]:
    target_url = urljoin(base_url, "/partialQuery/") 
    st.subheader("Partial recipe data")
    partial_recipe_data = st.multiselect("Choose the features that you want to provide:",
                                         options=list(recipe_data.keys()))
    if 'cultural_restriction' in partial_recipe_data:
        recipe_data['cultural_restriction'] = st.selectbox("Select the recipe cultural restriction:",
                                                    options=dfv.recipe_restriction) 
    if 'cultural_restriction' in partial_recipe_data:
        recipe_data['calories']=st.number_input("Introduce recipe calories:", min_value=0.0)
    if 'allergens' in partial_recipe_data:
        recipe_data['allergens'] =  st.selectbox(f"Please select the recipe allergens:",
                                options=list(dfv.allergies_queries_dict.keys()))
    if 'taste' in partial_recipe_data:
        recipe_data['taste'] =  st.selectbox(f"Please select recipe taste:", options=dfv.tastes)
    if 'price' in partial_recipe_data:
        recipe_data['price'] = st.number_input("Introduce recipe price (1= cheap, 3=expensive):", min_value=1.0, max_value=3.0, value=2.0) 
    if 'fiber' in partial_recipe_data:
        recipe_data['fiber'] = st.number_input("Introduce recipe fiber:", min_value=0.0)
    if 'fat' in partial_recipe_data:
        recipe_data['fat'] = st.number_input("Introduce recipe fat:", min_value=0.0)
    if 'protein' in partial_recipe_data:
        recipe_data['protein'] = st.number_input("Introduce recipe protein:", min_value=0.0)
    if 'carbohydrates' in partial_recipe_data:
        recipe_data['carbohydrates'] = st.number_input("Introduce recipe carbohydrates:", min_value=0.0)
    if 'ingredient_list' in partial_recipe_data: 
        recipe_data['ingredients'] = st.text_input("Introduce ingredients:")
    # ADD query to the the api
    data = {}
    data["profile"] = profile
    data["recipe"] = recipe_data
    data["context"] = context 
    st.write("Target url: %s" % target_url)
else:
    target_url = urljoin(base_url, "/checkCompatibility/")
    st.subheader("Recipe data", divider=True)
    st.write('Please complete the following recipe data:')
    recipe_data['cultural_restriction'] = st.selectbox("Select the recipe cultural restriction:",
                                                       options=dfv.recipe_restriction) 
    recipe_data['calories']=st.number_input("Introduce recipe calories:", min_value=0.0)
    recipe_data['allergens'] =  st.selectbox(f"Please select the recipe allergens:",
                                   options=list(dfv.allergies_queries_dict.keys()))
    recipe_data['taste'] =  st.selectbox(f"Please select recipe taste:", options=dfv.tastes)
    recipe_data['price'] = st.number_input("Introduce recipe price (1= cheap, 3=expensive):", min_value=1.0, max_value=3.0, value=2.0) 
    recipe_data['fiber'] = st.number_input("Introduce recipe fiber:", min_value=0.0)
    recipe_data['fat'] = st.number_input("Introduce recipe fat:", min_value=0.0)
    recipe_data['protein'] = st.number_input("Introduce recipe protein:", min_value=0.0)
    recipe_data['carbohydrates'] = st.number_input("Introduce recipe carbohydrates:", min_value=0.0) 
    recipe_data['ingredients'] = st.text_input("Introduce ingredients:")
    # ADD query to the the api
    data = {}
    data["profile"] = profile
    data["recipe_data"] = recipe_data
    data["context"] = context 
    st.write("Target url: %s" % target_url)

# show answer
if st.button("Predict"):
    st.write("Bottom clicked on")
    response = requests.post(target_url, json=data)
    st.write(response.json())

# show the references 
st.subheader("References", divider=True)
st.write("""
Victor H. Contreras, Michael Schumacher and Davide Calvaresi, Explanation of Deep Learning Models via Logic Rules Enhanced by Embeddings Analysis, and Probabilistic Models, in: Post-proceedings of the 6th International Workshop on EXplainable and TRAnsparent AI and Multi-Agent Systems, 2024
         """)