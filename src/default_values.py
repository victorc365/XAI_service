# dictionary allergy queries with synonyms
allergies_queries_dict = {'tree nuts': ['tree', 'nuts', 'nut', 'tree nuts'],
                          'wheat': ['wheat', 'grain', 'gluten'],
                          'eggs': ['eggs', 'egg'],
                          'soy': ['soy', 'soya', 'Glycine max'],
                          'fish': ['fish', 'salmon', 'seafood', 'tuna'],
                          'peanut': ['peanut', 'groundnut', 'mani'],
                          'shellfish': ['shellfish', 'clam', 'lobster', 'scallop', 'mollusk', 'snail'],
                          "cow's milk": ["cow's milk", "milk", "lactose"],
                          "NotAllergy": []}

user_allergies = ['NotAllergy',
  'wheat',
  'peanut',
  "cow's milk",
  'soy',
  'shellfish',
  'Multiple']

cultural_factors = {
    "vegan_observant": [True, False],
    "vegetarian_observant": [True, False],
    "halal_observant": [True, False],
    "kosher_observant": [True, False],
    "flexi_observant": [True, False],
    "NotRestriction": []
}

# Calorie distribution across meals
meals_calorie_dict = {"breakfast": 0.3,
                      "morning snacks": 0.05,
                      "afternoon snacks": 0.05,
                      "lunch": 0.4,
                      "dinner": 0.2
                      }

flexi_probabilities_dict = {
    "flexi_vegan": {
        "vegan_observant": 0.60,
        "vegetarian_observant": 0.20,
        "halal_observant": 0.10,
        "kosher_observant": 0.1,
        "None": 0.0
    },
    "flexi_vegetarian": {
        # no meaningful flexi for this class:
        # vegetarian -> vegan
        "vegan_observant": 0.00,
        "vegetarian_observant": 0.60,
        "halal_observant": 0.0,
        "kosher_observant": 0.1,
        "None": 0.30
    },
    "flexi_halal": {
        "vegan_observant": 0.1,
        "vegetarian_observant":  0.2,
        "halal_observant": 0.6,
        "kosher_observant": 0.0,
        "None": 0.1
    },
    "flexi_kosher": {
        "vegan_observant": 0.1,
        "vegetarian_observant": 0.1,
        "halal_observant": 0.1,
        "kosher_observant": 0.6,
        "None": 0.1
    }
}

place_proba_dict = {
    "restaurant": 0.3,
    "home": 0.5,
    "outdoor": 0.2
}

social_situation_proba_dict = {
    "alone": 0.3,
    "family": 0.3,
    "friends": 0.2,
    "colleagues": 0.2
}

time_options = {
    "now": 0.5,
    "in one hour": 1.0,
    "in two hours": 2.5,
    "other time": 0.0
}

user_entity = {
    "current_working_status": ["Half-time-worker", "Full-time-worker", "Self-employee", "Unemployed"],
    "marital_status": ["Single", "Married"],
    "life_style": ["Sedentary", "Lightly active", "Moderately active", "Very active"],
    "weight": [],
    "ethnicity": ["White", "Black", "Latino", "Asian"],
    "height": []
}

nutritional_goals = {"lose_weight", "maintain_fit", "gain_weight"}

age_dict = {
        "18-29": 0.10,
        "30-39": 0.10,
        "40-49": 0.10,
        "50-59": 0.20,
        "60-69": 0.20,
        "70-79": 0.10,
        "80-89": 0.10,
        "90-100": 0.10
    }

bmi_set = {"underweight", "healthy", "overweight", "obesity"}
tastes = ['sweet', 'sour', 'salty', 'bitter', 'umami']
recipe_restriction = ['vegan', 'vegetarian', 'halal', 'keto', 'kosher', 'meat-based', 'NotRestriction', 
                      'pescatarian', 'seafood-based']

safety_allergies = ['NotAllergy', 'wheat', 'peanut', "cow's milk", 'soy', 'shellfish', 'Multiple']
safety_allergens = ['legumes', 'Soybeans', 'NotAllergens', 'tree nuts', 'soy',
       'lactose', 'dairy', 'peanuts', 'garlic', 'dairy, gluten',
       'shellfish', 'Milk', 'seafood', 'gluten', 'eggs', 'lactose, eggs',
       'soy, peanuts', 'soy, gluten', 'wheat', 'tree nuts, gluten',
       'Wheat', 'Eggs', 'sesame', 'soy, wheat', 'wheat, dairy',
       'lactose, dairy', 'poultry', 'Tree nuts', 'Shellfish',
       'nightshade', 'tree nuts, dairy', 'seafood, lactose', 'meat',
       'sesame, soy, peanuts, gluten', 'peanuts, eggs',
       'shellfish, peanuts', 'soy, wheat, pork', 'Dairy',
       'wheat, tree nuts', 'Fish', 'legumes, dairy', 'soy, pork',
       'seafood, dairy', 'soy, shellfish', 'soy, shellfish, gluten',
       'soy, wheat, shellfish', 'legumes, corn', 'pork', 'black beans',
       'seafood, citrus', 'dairy, eggs', 'eggs, gluten',
       'pork, soy, gluten', 'sesame, soy, eggs', 'spices',
       'legumes, gluten', 'seafood, soy', 'soy, shellfish, eggs',
       'peanuts, gluten', 'shellfish, gluten', 'Peanuts',
       'lactose, tree nuts', 'cilantro', 'coconut', 'sesame, gluten',
       'soy, tree nuts', 'seafood, gluten', 'shellfish, dairy',
       'sesame, soy', 'wheat, tree nuts, dairy', 'mustard',
       'mustard, tree nuts', 'gluten, eggs', 'citrus',
       'gluten, soy, peanuts', 'sesame, eggs', 'corn', 'wheat, shellfish',
       'seafood, shellfish', 'pork, gluten', 'wheat, eggs',
       'shellfish, tree nuts', 'wheat, lactose', 'Crustacean shellfish',
       'soy, eggs', 'rice', 'rice, gluten']

meal_type_y=['veggie', 'Meat-based', 'vegan', 'vegetarian', 'smoothie',
       'grain-based', 'dessert', 'Vegetarian', 'lunch;dinner',
       'meat-based', 'NotInformation', 'Beverage', 'fruit-based',
       'Dessert', 'pasta', 'cheesy', 'breakfast', 'italian',
       'breakfast;lunch;dinner', 'Stuffing', 'Appetizer', 'seafood',
       'sushi', 'the meal type for this recipe is vegetarian.',
       'Fruit-based', 'fish',
       'this recipe can be classified as seafood-based.',
       'breakfast;dinner',
       'based on the ingredients provided, the meal type of the recipe would be "vegetarian."',
       'The meal type for this recipe is "veggie".', 'Mexican', 'Seafood',
       'Muffins', 'brunch', 'Basic', 'snack', 'Herb-based', 'indian',
       'sandwich', 'asian', 'protein-based', 'appetizer', 'cheese',
       'breakfast;lunch', 'Drink', 'bread',
       'The meal type of the recipe is grain-based.', 'dinner', 'lunch',
       'in this recipe, the meal type is vegetarian.', 'Grain-based',
       '\nMeat-based', 'salad', 'Biscuits', 'mexican', 'Bread-based',
       'Snack', 'Veggie', 'Breakfast',
       'the meal type for this recipe is meat-based.', 'fruit',
       'the meal type of this recipe is meat-based.', 'Baking', 'Vegan',
       'Spice-based', 'Condiment', 'Brunch', 'baked goods',
       'lactose-free', 'Kosher', 'Baked',
       'no information is given about the specific dietary restrictions or preferences, but based on the ingredients listed, the meal type for this recipe would be "protein-based."',
       'Seafood-based',
       'sorry, i am unable to generate the recipe based on the information provided. however, the meal type of the cheese stuffed mini bell peppers would typically be considered "vegetarian."',
       'This recipe is meat-based.', 'lebanese fatayer- meat-based',
       'veggie/vegetarian', 'pescatarian',
       'there is not enough information to determine the meal type of this recipe.',
       'meatless', 'the meal type of this recipe is "dessert".',
       'beverage', 'oatmeal', 'non-vegetarian', 'pizza', 'fish-based',
       'eggs', 'sandwich.', 'dairy', 'this recipe is meat-based.',
       'baking', 'egg-based', 'Candy', 'vegetarian.', 'Baked-Goods',
       'baklava', 'nut-based', 'sweet', 'Dairy', 'Cocktail', 'caprese',
       'toast', 'omelette', 'the meal type for this recipe is seafood.',
       'Alcoholic', 'Spicy', 'korean',
       'the meal type of this recipe is vegetarian.', 'plant-based',
       'coconut', 'bruschetta', 'fusion', 'the meal type is greek.',
       'kosher', 'halal', 'smoothie bowl', 'Pickles',
       'the meal type of the recipe is indian.',
       'keywords like "veggie" and "hummus" indicate that the meal type for this recipe is vegetarian.',
       'not specified', 'cheese-based',
       'plain, chocolate chip, raspberry', 'this recipe is vegetarian.',
       'pastry', 'unspecified', 'greek', 'bakery', 'breads',
       'the meal type of the recipe is "pastry".', 'spicy',
       'frozen dessert', 'savory', 'berry-yogurt parfait - vegetarian',
       'the meal types for the provided recipes are seafood-based or sushi-based.',
       'pasta.', 'the meal type of the recipe is "vegan".',
       'based on the ingredients listed, the meal type of this recipe would be vegetarian.',
       'meat', 'american', 'the meal type for this recipe is sushi.',
       'as the recipe includes both tofu or chicken, it can be considered as "meat-based".',
       'graubased', 'Protein', 'this recipe is a meat-based recipe.',
       'Sweet',
       'there is no specific indication of the meal type in the given recipe.',
       'the meal type of the recipe is "dessert."']



default_sample = {'BMI': ['healthy'], 
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
     'cluster':[2], 
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
     'weight': [65],
     'y_pred': [0]
     }