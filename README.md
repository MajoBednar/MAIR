# MAIR
For UU course Methods in AI Research

# Restaurant Recommendation Dialog System

This project implements a dialog system for restaurant recommendations using state transitions, dialog act classification, and preference extraction.

## Directory Structure

```
MAIR/
├── data/       # Contains restaurant_info.csv and dialog_acts.dat
├── models/     # Contains pretrained models (e.g., nn_full.pkl)
├── src/
│   ├── Baseline_systems.py
│   ├── bow.py
│   ├── classifier_evaluation.py
│   ├── data_management.py
│   ├── domain_terms.py
│   ├── extract_preferences.py
│   ├── ml_models.py
│   ├── Restaurant_lookup.py
│   ├── state_transition.py
│   ├── ui.py
│   └── __pycache__/
```

## Setup

1. **Install dependencies:**
    Required packages are:
    - pandas
    - scikit-learn
    - python-Levenshtein

2. **Data files:**
    - Place `restaurant_info.csv` and `dialog_acts.dat` in the `data/` folder.

## How to use

Run the dialog system from the `src` directory:
```
python state_transition.py
```

Follow the prompts to interact with the restaurant recommendation system.

## Main Components

- **state_transition.py**: Main dialog manager using state transitions and dialog acts.
- **extract_preferences.py**: Extracts user preferences (area, food, price) from utterances.
- **Restaurant_lookup.py**: Looks up restaurants matching user preferences.
- **domain_terms.py**: Contains domain-specific terms (areas, food types, price ranges).
- **data_management.py**: Handles data loading and saving.
- **bow.py**: Bag-of-words feature extraction.
- **ml_models.py**: Machine learning models for dialog act classification.
- **Baseline_systems.py**: Baseline classifiers and data splitting.
- **classifier_evaluation.py**: Evaluation scripts for classifiers.
- **ui.py**: (Optional) User interface components.

## Notes

- All user input is converted to lowercase for processing.
- Dialog acts are classified using a ML classifier.
- The system uses Levenshtein distance for keyword matching.

## Current Issues

- The system does not support "don't care" preferences.
- The system does not explicitly confirm the known preferences.
- The dialog act classifier sometimes misclassifies utterances.
- Does not yet state restaurant match immediately, needs preliminary. 
- Dialog system (mainly state transition function) needs more testing and improvement.

## Group Number & Members
group number: F3
members:
- Majo Bednár
- Lonneke Hormann
- Finn Joosten
- Jop Paro

