import re
from Levenshtein import distance

PRICE_RANGE_TERMS = ['cheap', 'moderate', 'expensive', 'any']
AREA_TERMS = ['east', 'west', 'north', 'south', 'centre', 'any']
FOOD_TERMS = [
    'british', 'modern european', 'italian', 'romanian', 'seafood', 'chinese',
    'steakhouse', 'asian oriental', 'french', 'portuguese', 'indian', 'spanish',
    'european', 'vietnamese', 'korean', 'thai', 'moroccan', 'swiss', 'fusion',
    'gastropub', 'tuscan', 'international', 'traditional', 'mediterranean',
    'polynesian', 'african', 'turkish', 'bistro', 'north american', 'australasian',
    'persian', 'jamaican', 'lebanese', 'cuban', 'japanese', 'catalan',
    'world', 'swedish', 'any'
]


def autocorrect(term: str, true_domain: list[str], max_correcting_dist: int) -> str | None:
    """
    Autocorrects the :param term to the closest term in the :param true_domain.
    This is done with Levenshtein edit distance. If the edit distance is > :param max_correcting_dist,
    or if the term is closer to any of the terms in the other domains, None is returned.
    """
    corrected_term = None
    true_smallest_edit_distance = 100
    smallest_edit_distance = 100

    for domain in (PRICE_RANGE_TERMS, AREA_TERMS, FOOD_TERMS):
        for domain_term in domain:
            edit_distance = distance(term, domain_term)
            # autocorrect to the closest word from wanted domain
            if domain == true_domain:
                if edit_distance <= max_correcting_dist and edit_distance < true_smallest_edit_distance:
                    corrected_term = domain_term
                    true_smallest_edit_distance = edit_distance
            # check if words from other domains are closer to the given term
            else:
                if edit_distance <= max_correcting_dist and edit_distance < smallest_edit_distance:
                    smallest_edit_distance = edit_distance

    # If a term is corrected, but it was actually a term from another domain
    if smallest_edit_distance < true_smallest_edit_distance:
        return None  # no preference was expressed for THIS domain
    
    # Safeguard: never autocorrect into 'any'
    if corrected_term == "any":
        return "any" if term == "any" else None

    return corrected_term


def keyword_fallback(utterance: str, term_domain: list[str]) -> str | None:
    """Traverses an utterance to find if a domain term is present. Returns that domain term."""
    for word in utterance.split():
        if word in term_domain:
            return word
    return None


def extract_price_range_pref(utterance: str, max_correcting_dist: int) -> str | None:
    """Extracts a preference for price range, or None."""
    preference = None
    # check for a suitable word occurring before 'restaurant'
    price_range = re.findall(r'\b(\w+)\s+restaurant', utterance)
    if price_range and price_range[0] != 'priced':
        preference = autocorrect(price_range[0], PRICE_RANGE_TERMS, max_correcting_dist) \
            if preference is None else preference
    # check for a suitable word occurring before 'price'
    price_range = re.findall(r'\b(\w+)\s+price', utterance)
    if price_range:
        preference = autocorrect(price_range[0], PRICE_RANGE_TERMS, max_correcting_dist) \
            if preference is None else preference
        
    # Direct keyword fallback, only if no explicit 'any' was given
    if preference is None:
        # Check if 'any' is explicitly tied to this category
        if re.search(r'\bany\s+(restaurant|food|area|place|spot|type|price|range)\b', utterance):
            preference = "any"
        else:
            # Only look for domain terms if no explicit 'any' was given
            preference = keyword_fallback(utterance, PRICE_RANGE_TERMS)
    return preference


def extract_area_pref(utterance: str, max_correcting_dist: int) -> str | None:
    """Extracts a preference for area, or None."""
    preference = None
    # check for a suitable word occurring before 'area'
    area = re.findall(r'\b(\w+)\s+area', utterance)
    if area:
        preference = autocorrect(area[0], AREA_TERMS, max_correcting_dist) if preference is None else preference
    # check for a suitable word occurring after 'in the'
    area = re.findall(r'in\s+the\s+(\w+)\b', utterance)
    if area:
        preference = autocorrect(area[0], AREA_TERMS, max_correcting_dist) if preference is None else preference

    # Direct keyword fallback, only if no explicit 'any' was given
    if preference is None:
        # Check if 'any' is explicitly tied to this category
        if re.search(r'\bany\s+(restaurant|food|area|place|spot|type|price|range)\b', utterance):
            preference = "any"
        else:
            # Only look for domain terms if no explicit 'any' was given
            preference = keyword_fallback(utterance, AREA_TERMS)
    return preference


def extract_food_pref(utterance: str, max_correcting_dist: int) -> str | None:
    """Extracts a preference for food, or None."""
    preference = None
    # check for a suitable word occurring before 'food'
    food = re.findall(r'\b(\w+)\s+food', utterance)
    if food:
        preference = autocorrect(food[0], FOOD_TERMS, max_correcting_dist) if preference is None else preference
    # check for a suitable word occurring before 'restaurant'
    food = re.findall(r'\b(\w+)\s+restaurant', utterance)
    if food and food[0] != 'priced':
        preference = autocorrect(food[0], FOOD_TERMS, max_correcting_dist) if preference is None else preference

    # Direct keyword fallback, only if no explicit 'any' was given
    if preference is None:
        # Check if 'any' is explicitly tied to this category
        if re.search(r'\bany\s+(restaurant|food|area|place|spot|type|price|range)\b', utterance):
            preference = "any"
        else:
            # Only look for domain terms if no explicit 'any' was given
            preference = keyword_fallback(utterance, FOOD_TERMS)
    return preference


def extract_preferences(utterance: str, max_correcting_dist: int = 3) -> tuple:
    """
    Returns preferences for price range, area and food.
    A preference for each category must be in the term domain of that category, or 'any'.
    If no preference for that category was expressed, return None.
    """
    utterance = utterance.lower()
    price_range_pref = extract_price_range_pref(utterance, max_correcting_dist)
    area_pref = extract_area_pref(utterance, max_correcting_dist)
    food_pref = extract_food_pref(utterance, max_correcting_dist)
    return price_range_pref, area_pref, food_pref


def extract_additional_preference(utterance: str) -> str | None:
    """Extract additional preferences. Also check if negation was present."""
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
    """Testing preference extraction."""
    user_input = ''
    while user_input != 'q()':
        user_input = input('Type your preferences for a restaurant\n')
        print(extract_preferences(user_input))
        print(extract_additional_preference(user_input))


if __name__ == '__main__':
    main()
