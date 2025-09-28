import re
from Levenshtein import distance
from domain_terms import PRICE_RANGE_TERMS, AREA_TERMS, FOOD_TERMS


def autocorrect(term: str, true_domain: list[str]):
    """
    Autocorrects the term parameter to the closest term in the domain parameter.
    This is done with Levenshtein edit distance. If the edit distance is > 3, 'unknown_XXX' is returned,
    XXX being the term in question.
    """
    corrected_term = None
    true_smallest_edit_distance = 100
    smallest_edit_distance = 100
    for domain in (PRICE_RANGE_TERMS, AREA_TERMS, FOOD_TERMS):
        for domain_term in domain:
            edit_distance = distance(term, domain_term)
            # print(term, domain_term, edit_distance)
            if domain == true_domain:
                if edit_distance <= 3 and edit_distance < true_smallest_edit_distance:
                    corrected_term = domain_term
                    true_smallest_edit_distance = edit_distance
            else:
                if edit_distance <= 3 and edit_distance < smallest_edit_distance:
                    smallest_edit_distance = edit_distance

    # if a term is corrected, but it was actually a term from another domain
    if smallest_edit_distance < true_smallest_edit_distance:
        return None  # no preference was expressed for THIS domain

    if corrected_term is None:
        return 'unknown_' + term
    return corrected_term


def extract_price_range_pref(utterance: str):
    preference = None
    price_range = re.findall(r'\b(\w\w+)\s+restaurant', utterance)
    if price_range and price_range[0] != 'priced':
        preference = autocorrect(price_range[0], PRICE_RANGE_TERMS) if preference is None else preference
    price_range = re.findall(r'\b(\w\w+)\s+price', utterance)
    if price_range:
        preference = autocorrect(price_range[0], PRICE_RANGE_TERMS) if preference is None else preference
    return preference


def extract_area_pref(utterance: str):
    preference = None
    area = re.findall(r'\b(\w\w+)\s+area', utterance)
    if area:
        preference = autocorrect(area[0], AREA_TERMS) if preference is None else preference
    area = re.findall(r'in\s+the\s+(\w\w+)\b', utterance)
    if area:
        preference = autocorrect(area[0], AREA_TERMS) if preference is None else preference
    return preference


def extract_food_pref(utterance: str):
    preference = None
    food = re.findall(r'\b(\w\w+)\s+food', utterance)
    if food:
        preference = autocorrect(food[0], FOOD_TERMS) if preference is None else preference
    food = re.findall(r'\b(\w\w+)\s+restaurant', utterance)
    if food and food[0] != 'priced':
        preference = autocorrect(food[0], FOOD_TERMS) if preference is None else preference
    return preference


def extract_preferences(utterance: str):
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
    price_range_pref = extract_price_range_pref(utterance)
    area_pref = extract_area_pref(utterance)
    food_pref = extract_food_pref(utterance)
    return price_range_pref, area_pref, food_pref


def main():
    user_input = ''
    while user_input != 'q()':
        user_input = input('Type your preferences for a restaurant\n')
        print(extract_preferences(user_input))


if __name__ == '__main__':
    main()
