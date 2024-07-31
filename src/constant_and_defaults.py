# entity features
user_features_list = ['nutrition_goal', 'clinical_gender', 'age_range',
                      'life_style', 'weight', 'height',
                      'projected_daily_calories', 'current_daily_calories',
                      'cultural_factor', 'allergy',
                      'current_working_status', 'marital_status', 'ethnicity',
                      'BMI', 'next_BMI']
#food_features = ['recipe_name', 'recipe_raw_text', 'meal_type_y',
#      'food_cultural_class', 'calories', 'allergens', 'recipeId']
food_features = ['cultural_restriction', 'calories', 'allergens', 'taste', 'price',  'fiber', 'fat', 'protein', 'carbohydrates'] #'recipeId',
context_features  = ['day_number', 'meal_type_y','time_of_meal_consumption', 'place_of_meal_consumption', 'social_situation_of_meal_consumption']
label_list = ['label']