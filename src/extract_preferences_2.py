import re
from Levenshtein import distance

logger = None  # to be initialized externally

PRICE_RANGE_TERMS = ['cheap', 'moderate', 'expensive', 'moderately']
AREA_TERMS = ['east', 'west', 'north', 'south', 'centre', 'center']
FOOD_TERMS = [
    'british', 'modern european', 'italian', 'romanian', 'seafood', 'chinese',
    'steakhouse', 'asian oriental', 'french', 'portuguese', 'indian', 'spanish',
    'european', 'vietnamese', 'korean', 'thai', 'moroccan', 'swiss', 'fusion',
    'gastropub', 'tuscan', 'international', 'traditional', 'mediterranean',
    'polynesian', 'african', 'turkish', 'bistro', 'north american', 'australasian',
    'persian', 'jamaican', 'lebanese', 'cuban', 'japanese', 'catalan',
    'world', 'swedish'
]

OTHER_TERMS = []  # any??? TODO: add words

VOCAB = PRICE_RANGE_TERMS + AREA_TERMS + FOOD_TERMS + OTHER_TERMS


def autocorrect(term: str, max_correcting_dist: int) -> str:
    corrected_term = None
    smallest_edit_distance = 100

    for word in VOCAB:
        edit_distance = distance(term, word)
        if edit_distance < smallest_edit_distance and edit_distance <= max_correcting_dist:
            corrected_term = word
            smallest_edit_distance = edit_distance

    if corrected_term and corrected_term != term and logger is not None:
        logger.log_autocorrect(term, corrected_term, term)
        return corrected_term
    return term


def autocorrect_sentence(sentence: str, max_correcting_dist: int) -> str:
    words = re.findall(r"[a-zA-Z']+", sentence)
    corrected_words = [
        autocorrect(word, max_correcting_dist) for word in words
    ]

    # Reconstruct sentence while preserving spacing/punctuation
    corrected_sentence = sentence
    for original, corrected in zip(words, corrected_words):
        corrected_sentence = re.sub(
            r'\b' + re.escape(original) + r'\b', corrected, corrected_sentence, 1
        )
    return corrected_sentence


def keyword_fallback(utterance: str, term_domain: list[str]) -> str | None:
    """Traverses an utterance to find if a domain term is present. Returns that domain term."""
    for word in utterance.split():
        if word in term_domain:
            return word
    return None


def extract_price_range_pref(utterance: str) -> str | None:
    preference = keyword_fallback(utterance, PRICE_RANGE_TERMS)
    if preference:
        if preference == 'moderately':
            return 'moderate'
        return preference

    if re.search(r'\bany\s+(price|range|restaurant|place|spot)\b', utterance):
        return 'any'
    return None


def extract_area_pref(utterance: str) -> str | None:
    preference = keyword_fallback(utterance, AREA_TERMS)
    if preference:
        if preference == 'center':
            return 'centre'
        return preference

    if re.search(r'\bany\s+(area|restaurant|place|spot)\b', utterance):
        return 'any'
    return None


def extract_food_pref(utterance: str) -> str | None:
    preference = keyword_fallback(utterance, FOOD_TERMS)
    if preference:
        return preference

    if re.search(r'\bany\s+(food|cuisine|type|restaurant|place|spot)\b', utterance):
        return 'any'
    return None


def extract_preferences(utterance: str, max_correcting_dist: int) -> tuple:
    utterance = utterance.lower()
    utterance = autocorrect_sentence(utterance, max_correcting_dist)
    if bool(re.fullmatch(r"\W*any\W*", utterance.strip().lower())):
        return 'any', 'any', 'any'

    price_range_pref = extract_price_range_pref(utterance)
    area_pref = extract_area_pref(utterance)
    food_pref = extract_food_pref(utterance)
    return price_range_pref, area_pref, food_pref
