from extract_preferences import extract_preferences
from Restaurant_lookup import restaurant_lookup
from ml_models import MLModel, MLP
import sys
import pandas as pd
import os
import random

# Load a pretrained dialog act classifier
sys.modules['__main__'].MLP = MLP  # Ensure MLP is available for unpickling
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'nn_full.pkl')
dialog_act_classifier = MLModel.load(model_path)

SYSTEM_UTTERANCES = {
    "welcome": "Welcome to the restaurant recommendation system! How can I assist you today?",
    "ask_preferences": "Please tell me your preferences (area, food type, price range).",
    "ask_area": "Which area are you interested in?",
    "ask_food": "What type of food would you like?",
    "ask_price": "What price range are you looking for?",
    "confirm_area": "You want a restaurant in the {area} area, correct?",
    "confirm_food": "You want {food} food, correct?",
    "confirm_price": "You want a {price} restaurant, correct?",
    "no_match": "Sorry, no restaurant matches your preferences. Would you like to try different preferences?",
    "suggest_restaurant": "I suggest: {restaurant}. Would you like more information or another suggestion?",
    "provide_info": "Here is the information you requested about {restaurant}.", #TODO
    "provide_postcode": "The postcode for {restaurant} is {postcode}.", #TODO
    "provide_phone": "The phone number for {restaurant} is {phone}.", #TODO
    "provide_address": "The address for {restaurant} is {addr}.", #TODO
    "ask_additional_preferences": "Do you have any additional preferences?", #TODO (1c)
    "goodbye": "Goodbye!",
    "clarify": "Sorry, I didn't understand. Could you please rephrase?",
}

CONFIG = {
    "levenshtein_dist": 3
}

def nextstate(currentstate, context, utterance, restaurant_df):
    """
    This function implements the state transition diagram for the restaurant recommendation dialogue system.
    It takes the current state, context, user utterance, and restaurant dataframe as input,
    and returns the next state, updated context, and system utterance.
    """
    utterance = utterance.lower().strip()
    dialog_act = dialog_act_classifier.predict_sentence(utterance)[0]


    # State 1a: Welcome
    if currentstate == "welcome":
        if dialog_act == "hello":
            return "ask_preferences", context, SYSTEM_UTTERANCES["ask_preferences"]
        elif dialog_act == "inform":
            price, area, food = extract_preferences(utterance, CONFIG["levenshtein_dist"])
            if area: context['area'] = area
            if food: context['food'] = food
            if price: context['price'] = price
            if not context.get('area'):
                return "ask_area", context, SYSTEM_UTTERANCES["ask_area"]
            elif not context.get('food'):
                return "ask_food", context, SYSTEM_UTTERANCES["ask_food"]
            elif not context.get('price'):
                return "ask_price", context, SYSTEM_UTTERANCES["ask_price"]
            else:
                return "suggest_restaurant", context, None
        elif dialog_act == "bye":
            return "goodbye", context, SYSTEM_UTTERANCES["goodbye"]
        else:
            return "welcome", context, SYSTEM_UTTERANCES["welcome"]

    # State 1b: Ask Preferences
    if currentstate == "ask_preferences":
        price, area, food = extract_preferences(utterance, CONFIG["levenshtein_dist"])
        if area: context['area'] = area
        if food: context['food'] = food
        if price: context['price'] = price
        if not context.get('area'):
            return "ask_area", context, SYSTEM_UTTERANCES["ask_area"]
        elif not context.get('food'):
            return "ask_food", context, SYSTEM_UTTERANCES["ask_food"]
        elif not context.get('price'):
            return "ask_price", context, SYSTEM_UTTERANCES["ask_price"]
        else:
            return "suggest_restaurant", context, None

    # State 2: Ask for area preference
    if currentstate == "ask_area":
        price, area, food = extract_preferences(utterance, CONFIG["levenshtein_dist"])
        if area:
            context['area'] = area
            if not context.get('food'):
                return "ask_food", context, SYSTEM_UTTERANCES["ask_food"]
            elif not context.get('price'):
                return "ask_price", context, SYSTEM_UTTERANCES["ask_price"]
            else:
                return "suggest_restaurant", context, None
        else:
            return "ask_area", context, SYSTEM_UTTERANCES["ask_area"]

    # State 3: Ask for food preference
    if currentstate == "ask_food":
        price, area, food = extract_preferences(utterance, CONFIG["levenshtein_dist"])
        if food:
            context['food'] = food
            if not context.get('price'):
                return "ask_price", context, SYSTEM_UTTERANCES["ask_price"]
            else:
                return "suggest_restaurant", context, None
        else:
            return "ask_food", context, SYSTEM_UTTERANCES["ask_food"]

    # State 4: Ask for price preference
    if currentstate == "ask_price":
        price, area, food = extract_preferences(utterance, CONFIG["levenshtein_dist"])
        if price:
            context['price'] = price
            return "suggest_restaurant", context, None
        else:
            return "ask_price", context, SYSTEM_UTTERANCES["ask_price"]

    # State 6: Suggest Restaurant
    if currentstate == "suggest_restaurant":
        price = context.get('price')
        area = context.get('area')
        food = context.get('food')
        matches = restaurant_lookup(restaurant_df, price, area, food)
        if matches:
            chosen = random.choice(matches)
            context['suggested'] = chosen
            context['alternatives'] = [m for m in matches if m != chosen]
            return "await_user_response", context, SYSTEM_UTTERANCES["suggest_restaurant"].format(restaurant=chosen)
        else:
            context['suggested'] = None
            context['alternatives'] = []
            return "no_match", context, SYSTEM_UTTERANCES["no_match"]

    # State 5: Ask other preferences
    if currentstate == "no_match":
        context['area'] = None
        context['food'] = None
        context['price'] = None
        return "ask_preferences", context, SYSTEM_UTTERANCES["ask_preferences"]

    # Awaiting user response
    if currentstate == "await_user_response":
        if dialog_act == "request":
            #TODO: classify what info is requested (postcode, phone, address) and chose provide_info, provide_postcode, provide_phone, provide_address accordingly.
            return "provide_info", context, SYSTEM_UTTERANCES["provide_info"].format(restaurant=context['suggested'])
        elif dialog_act in ["reqalts", "reqmore"]:
            if context.get('alternatives'):
                chosen = random.choice(context['alternatives'])
                context['suggested'] = chosen
                context['alternatives'].remove(chosen)
                return "await_user_response", context, SYSTEM_UTTERANCES["suggest_restaurant"].format(restaurant=chosen)
            else:
                return "no_match", context, SYSTEM_UTTERANCES["no_match"]
        elif dialog_act in ["thankyou", "bye"]:
            return "goodbye", context, SYSTEM_UTTERANCES["goodbye"]
        elif dialog_act == "restart":
            context = {'area': None, 'food': None, 'price': None, 'suggested': None, 'alternatives': []}
            return "welcome", context, SYSTEM_UTTERANCES["welcome"]
        else:
            return "await_user_response", context, SYSTEM_UTTERANCES["clarify"]

    # State 7: Provide Information TODO: expand with postcode, phone, address
    if currentstate == "provide_info":
        #TODO: classify what info is requested (postcode, phone, address)
        return "await_user_response", context, SYSTEM_UTTERANCES["clarify"]

    if currentstate == "goodbye":
        return "goodbye", context, SYSTEM_UTTERANCES["goodbye"]

    return "welcome", context, SYSTEM_UTTERANCES["clarify"]

def main():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'restaurant_info.csv')
    restaurant_df = pd.read_csv(data_path)
    state = "welcome"
    context = {'area': None, 'food': None, 'price': None, 'suggested': None, 'alternatives': []}
    print(SYSTEM_UTTERANCES["welcome"])
    while True:
        user_input = input("> ")
        if user_input.strip().lower() in ["quit", "exit", "q()", "bye"]:
            print(SYSTEM_UTTERANCES["goodbye"])
            break
        state, context, sysutt = nextstate(state, context, user_input, restaurant_df)
        if sysutt:
            print(sysutt)

if __name__ == '__main__':
    main()