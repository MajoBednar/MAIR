import re
from Levenshtein import distance
from domain_terms import PRICE_RANGE_TERMS, AREA_TERMS, FOOD_TERMS, CROWDEDNESS_TERMS, LENGTHOFSTAY_TERMS


def autocorrect(term: str, true_domain: list[str], max_correcting_dist):
    """
    Autocorrects the term parameter to the closest term in the domain parameter.
    This is done with Levenshtein edit distance. If the edit distance is > :param max_correcting_dist,
    'unknown_XXX' is returned, XXX being the term in question.
    """
    corrected_term = None
    true_smallest_edit_distance = 100
    smallest_edit_distance = 100
    for domain in (PRICE_RANGE_TERMS, AREA_TERMS, FOOD_TERMS, CROWDEDNESS_TERMS, LENGTHOFSTAY_TERMS):
        for domain_term in domain:
            edit_distance = distance(term, domain_term)
            # print(term, domain_term, edit_distance)
            if domain == true_domain:
                if edit_distance <= max_correcting_dist and edit_distance < true_smallest_edit_distance:
                    corrected_term = domain_term
                    true_smallest_edit_distance = edit_distance
            else:
                if edit_distance <= max_correcting_dist and edit_distance < smallest_edit_distance:
                    smallest_edit_distance = edit_distance

    # if a term is corrected, but it was actually a term from another domain
    if smallest_edit_distance < true_smallest_edit_distance:
        return None  # no preference was expressed for THIS domain

    if corrected_term is None:
        return 'unknown_' + term
    return corrected_term


def extract_price_range_pref(utterance: str, max_correcting_dist):
    preference = None
    price_range = re.findall(r'\b(\w\w+)\s+restaurant', utterance)
    if price_range and price_range[0] != 'priced':
        preference = autocorrect(price_range[0], PRICE_RANGE_TERMS, max_correcting_dist) \
            if preference is None else preference
    price_range = re.findall(r'\b(\w\w+)\s+price', utterance)
    if price_range:
        preference = autocorrect(price_range[0], PRICE_RANGE_TERMS, max_correcting_dist) \
            if preference is None else preference
        
    # NEW: direct keyword fallback
    if preference is None:
        for word in utterance.split():
            candidate = autocorrect(word, PRICE_RANGE_TERMS, max_correcting_dist)
            if candidate not in (None, f"unknown_{word}"):
                preference = candidate
                break
    return preference


def extract_area_pref(utterance: str, max_correcting_dist):
    preference = None
    area = re.findall(r'\b(\w\w+)\s+area', utterance)
    if area:
        preference = autocorrect(area[0], AREA_TERMS, max_correcting_dist) if preference is None else preference
    area = re.findall(r'in\s+the\s+(\w\w+)\b', utterance)
    if area:
        preference = autocorrect(area[0], AREA_TERMS, max_correcting_dist) if preference is None else preference

    # NEW: direct keyword fallback
    if preference is None:
        for word in utterance.split():
            candidate = autocorrect(word, AREA_TERMS, max_correcting_dist)
            if candidate not in (None, f"unknown_{word}"):
                preference = candidate
                break
    return preference


def extract_food_pref(utterance: str, max_correcting_dist):
    preference = None
    food = re.findall(r'\b(\w\w+)\s+food', utterance)
    if food:
        preference = autocorrect(food[0], FOOD_TERMS, max_correcting_dist) if preference is None else preference
    food = re.findall(r'\b(\w\w+)\s+restaurant', utterance)
    if food and food[0] != 'priced':
        preference = autocorrect(food[0], FOOD_TERMS, max_correcting_dist) if preference is None else preference

    # NEW: direct keyword fallback
    if preference is None:
        for word in utterance.split():
            candidate = autocorrect(word, FOOD_TERMS, max_correcting_dist)
            if candidate not in (None, f"unknown_{word}"):
                preference = candidate
                break
    return preference


def extract_preferences(utterance: str, max_correcting_dist=3):
    """
    Returns preferences for price range, area and food.
    If no preference for that category was expressed, return empty string ('').
    A preference for each category must be in the term domain of that category, or 'any'.
    If a preference term is not in the corresponding term domain, an 'unknown_XXX'
    preference is returned (XXX being the preference term used).
    (In case of an unknown preference, the dialog system should re-ask for that preference,
    stating that such XXX preference is not possible.)
    """
    utterance = utterance.lower()
    price_range_pref = extract_price_range_pref(utterance, max_correcting_dist)
    area_pref = extract_area_pref(utterance, max_correcting_dist)
    food_pref = extract_food_pref(utterance, max_correcting_dist)
    return price_range_pref, area_pref, food_pref


def extract_additional_preference(utterance: str):
    if re.findall(r'[Nn][Oo]', utterance) or re.findall(r'[Dd][Oo][Nn].?[Tt]', utterance):
        negation = 'not '
    else:
        negation = ''
    preference = re.findall(r'[Tt]ouristic', utterance)
    if preference:
        return negation + 'touristic'
    preference = re.findall(r'[Aa]ssigned\s+seats', utterance)
    if preference:
        return negation + 'assigned seats'
    preference = re.findall(r'[Cc]hildren', utterance)
    if preference:
        return negation + 'children'
    preference = re.findall(r'[Rr]omantic', utterance)
    if preference:
        return negation + 'romantic'
    return None


def main():
    user_input = ''
    while user_input != 'q()':
        user_input = input('Type your preferences for a restaurant\n')
        print(extract_preferences(user_input))
        print(extract_additional_preference(user_input))


if __name__ == '__main__':
    main()
