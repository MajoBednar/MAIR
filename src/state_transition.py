from extract_preferences import extract_preferences, extract_additional_preference
from infer_properties import InferredProperties, preference_reasoning
from Restaurant_lookup import restaurant_lookup
from baseline_systems import BaselineRules
from ml_models import MLModel, MLP
import sys
import pandas as pd
import os
import random

CONFIG = {
    "levenshtein_dist": 3,
    "use_confirmation": True,
    "caps_output": False,
    "use_baseline_dialog_act_recognition": False
}

# Load a pretrained dialog act classifier
if CONFIG["use_baseline_dialog_act_recognition"]:
    dialog_act_classifier = BaselineRules()

else:
    sys.modules['__main__'].MLP = MLP  # Ensure MLP is available for unpickling
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'nn_full.pkl')
    dialog_act_classifier = MLModel.load(model_path)

SYSTEM_UTTERANCES = {
    "welcome": "Welcome to the restaurant recommendation system! \n"
    "I can help you find a restaurant based on your preferences for area, food type, and price range.\n"
    "If you don't care about a certain preference, just say 'any'.\n"
    "Lets get started! Please tell me your preferences.",
    "ask_preferences": "Please tell me your preferences (area, food type, price range).",
    "ask_area": "Which area are you interested in?",
    "ask_food": "What type of food would you like?",
    "ask_price": "What price range are you looking for?",
    "confirm_area": "You want a restaurant in the {area} area, correct?",
    "confirm_food": "You want {food} food, correct?",
    "confirm_price": "You want a {price} restaurant, correct?",
    "no_match": "Sorry, no restaurant matches your preferences. Would you like to try different preferences?",
    "provide_info": "Here is the information you requested about {restaurant}.",
    "provide_postcode": "The postcode for {restaurant} is {postcode}.", 
    "provide_phone": "The phone number for {restaurant} is {phone}.", 
    "provide_address": "The address for {restaurant} is {addr}.",
    "suggest_restaurant": "I suggest: {restaurant}. Would you like more information or another suggestion?",
    "ask_additional_preferences": "There are multiple restaurants to choose from.\n"
                                  "What is your additional preference (touristic/assigned seats/children/romantic)?",
    "goodbye": "Goodbye!",
    "clarify": "Sorry, I didn't understand. Could you please rephrase?",
}

def nextstate(currentstate, context, utterance, restaurant_df):
    """
    This function implements the state transition diagram for the restaurant recommendation dialogue system.
    It takes the current state, context, user utterance, and restaurant dataframe as input,
    and returns the next state, updated context, and system utterance.
    """
    utterance = utterance.lower().strip()
    dialog_act = dialog_act_classifier.predict_sentence(utterance)[0]
    # Output for debug
    print('Dialog act: ', dialog_act)
    print('Current state is: ', currentstate)

    # State 1: Welcome
    if currentstate == "welcome":
        if dialog_act == "hello":
            return "ask_preferences", context, SYSTEM_UTTERANCES["ask_preferences"]
        elif dialog_act == "inform":
            price, area, food = extract_preferences(utterance, CONFIG["levenshtein_dist"])
            if area: 
                context['area'] = area
                if CONFIG.get("use_confirmation", False):
                    return "confirm_area", context, SYSTEM_UTTERANCES["confirm_area"].format(area=area)
            if food: 
                context['food'] = food
                if CONFIG.get("use_confirmation", False):
                    return "confirm_food", context, SYSTEM_UTTERANCES["confirm_food"].format(food=food)
            if price: 
                context['price'] = price
                if CONFIG.get("use_confirmation", False):
                    return "confirm_price", context, SYSTEM_UTTERANCES["confirm_price"].format(price=price)
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

    # State 5: Ask Preferences
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
            if CONFIG.get("use_confirmation", False):
                return "confirm_area", context, SYSTEM_UTTERANCES["confirm_area"].format(area=area)
            else:
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
            if CONFIG.get("use_confirmation", False):
                return "confirm_food", context, SYSTEM_UTTERANCES["confirm_food"].format(food=food)
            else:
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
            if CONFIG.get("use_confirmation", False):
                return "confirm_price", context, SYSTEM_UTTERANCES["confirm_price"].format(price=price)
            else:
                return "suggest_restaurant", context, None
        else:
            return "ask_price", context, SYSTEM_UTTERANCES["ask_price"]

    # State 6: Suggest Restaurant
    if currentstate == "suggest_restaurant":
        price = context.get('price')
        area = context.get('area')
        food = context.get('food')
        matches = restaurant_lookup(restaurant_df, price, area, food)
        if len(matches) == 1:
            context['suggested'] = matches
            return "await_user_response", context, SYSTEM_UTTERANCES["suggest_restaurant"].format(restaurant=matches)
        if len(matches) > 1:
            context['alternatives'] = [m for m in matches]
            return "ask_additional_preferences", context, SYSTEM_UTTERANCES["ask_additional_preferences"]
        else:
            context['suggested'] = None
            context['alternatives'] = []
            return "no_match", context, SYSTEM_UTTERANCES["no_match"]

    # No restaurant match available: reset the preferences and ask again (go to state 5)
    if currentstate == "no_match":
        context['area'] = None
        context['food'] = None
        context['price'] = None
        return "ask_preferences", context, SYSTEM_UTTERANCES["ask_preferences"]

    # State 8: Ask for additional preferences  
    if currentstate == "ask_additional_preferences":
        additional_preference = extract_additional_preference(utterance)
        restaurant_alternatives = context["alternatives"]
        context["alternatives"] = []
        for restaurant in restaurant_alternatives:
            inferred_properties = InferredProperties(restaurant, restaurant_df)
            preference_satisfied = inferred_properties.is_preference_satisfied(additional_preference)
            if preference_satisfied:
                context["alternatives"].append(restaurant)
        if context["alternatives"]:
            chosen = context["alternatives"].pop(random.randrange(len(context["alternatives"])))
            context['suggested'] = chosen
            chosen += preference_reasoning(additional_preference)
            return "await_user_response", context, SYSTEM_UTTERANCES["suggest_restaurant"].format(restaurant=chosen)
        else:
            context['suggested'] = None
            context['alternatives'] = []
            return "no_match", context, SYSTEM_UTTERANCES["no_match"]

    # Awaiting user response..
    if currentstate == "await_user_response":
        if dialog_act == "request":
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

    # State 7: Provide Information
    if currentstate == "provide_info":
        # Classify what info is requested (postcode, phone, address)
        requested_info = None
        if "postcode" in utterance or "zip" in utterance:
            requested_info = "postcode"
        elif "phone" in utterance or "number" in utterance or "contact" in utterance:
            requested_info = "phone"
        elif "address" in utterance or "location" in utterance:
            requested_info = "address"

        restaurant = context.get('suggested')
        if isinstance(restaurant, list):  # If suggested is a list, pick the first
            restaurant = restaurant[0] if restaurant else None

        if restaurant is not None and requested_info:
            # Find the restaurant row in the dataframe
            row = restaurant_df[restaurant_df['name'].str.lower() == restaurant.lower()]
            if not row.empty:
                if requested_info == "postcode":
                    postcode = row.iloc[0].get('postcode', 'unknown')
                    return "await_user_response", context, SYSTEM_UTTERANCES["provide_postcode"].format(restaurant=restaurant, postcode=postcode)
                elif requested_info == "phone":
                    phone = row.iloc[0].get('phone', 'unknown')
                    return "await_user_response", context, SYSTEM_UTTERANCES["provide_phone"].format(restaurant=restaurant, phone=phone)
                elif requested_info == "address":
                    addr = row.iloc[0].get('address', 'unknown')
                    return "await_user_response", context, SYSTEM_UTTERANCES["provide_address"].format(restaurant=restaurant, addr=addr)
            # If restaurant not found or info missing
            return "await_user_response", context, f"Sorry, I couldn't find the {requested_info} for {restaurant}."
        else:
            # If info type not recognized or restaurant missing
            return "await_user_response", context, "Sorry, I couldn't understand what information you requested. Please ask for postcode, phone, or address."
    if currentstate == "goodbye":
        return "goodbye", context, SYSTEM_UTTERANCES["goodbye"]

    # State 12: Confirm Area
    if currentstate == "confirm_area":
        if dialog_act == "affirm":
            if not context.get('food'):
                return "ask_food", context, SYSTEM_UTTERANCES["ask_food"]
            elif not context.get('price'):
                return "ask_price", context, SYSTEM_UTTERANCES["ask_price"]
            else:
                return "suggest_restaurant", context, None
        elif dialog_act == "deny":
            context['area'] = None
            return "ask_area", context, SYSTEM_UTTERANCES["ask_area"]
        else:
            return "confirm_area", context, SYSTEM_UTTERANCES["confirm_area"].format(area=context['area'])
        
    # State 13: Confirm Food
    if currentstate == "confirm_food":
        if dialog_act == "affirm":
            if not context.get('price'):
                return "ask_price", context, SYSTEM_UTTERANCES["ask_price"]
            else:
                return "suggest_restaurant", context, None
        elif dialog_act == "deny":
            context['food'] = None
            return "ask_food", context, SYSTEM_UTTERANCES["ask_food"]
        else:
            return "confirm_food", context, SYSTEM_UTTERANCES["confirm_food"].format(food=context['food'])
        
    # State 14: Confirm Price
    if currentstate == "confirm_price":
        if dialog_act == "affirm":
            return "suggest_restaurant", context, None
        elif dialog_act == "deny":
            context['price'] = None
            return "ask_price", context, SYSTEM_UTTERANCES["ask_price"]
        else:
            return "confirm_price", context, SYSTEM_UTTERANCES["confirm_price"].format(price=context['price'])

    return "welcome", context, SYSTEM_UTTERANCES["clarify"]

def main():
    # Initialize restaurant dataframe and state transition parameters
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'restaurant_info.csv')
    restaurant_df = pd.read_csv(data_path)
    state = "welcome"
    context = {'area': None, 'food': None, 'price': None, 'suggested': None, 'alternatives': []}
    # Print welcome in correct casing
    welcome = SYSTEM_UTTERANCES["welcome"]
    print(welcome.upper() if CONFIG.get("caps_output", False) else welcome)

    while True:
        if state in ["suggest_restaurant", "no_match"]:
            user_input = "" # Skip user input to automatically suggest or handle no match
        else:
            user_input = input("> ")
            if user_input.strip().lower() in ["quit", "exit", "q()", "bye"]:
                goodbye = SYSTEM_UTTERANCES["goodbye"]
                print(goodbye.upper() if CONFIG.get("caps_output", False) else goodbye)
                break

        state, context, sysutt = nextstate(state, context, user_input, restaurant_df)
        print(context) # For debugging
        if sysutt:
            print(sysutt.upper() if CONFIG.get("caps_output", False) else sysutt)

if __name__ == '__main__':
    main()