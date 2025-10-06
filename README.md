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
```
        pip install -r requirements.txt
```

## How to use

Run the dialog system from the `src` directory:
```
python state_transition.py
```

Follow the prompts to interact with the restaurant recommendation system.

## Main Components

- **state_transition.py**: Main dialog manager using state transitions and dialog acts.
- **extract_preferences.py**: Extracts user preferences (area, food, price) from utterances.
- **restaurant_lookup.py**: Looks up restaurants matching user preferences.
- **infer_properties**: Infer other properties (romantic, touristic), based on prespecified rules.
- **data_management.py**: Handles data loading and saving.
- **bag_of_words.py**: Bag-of-words feature extraction.
- **ml_models.py**: Machine learning models for dialog act classification.
- **baseline_systems.py**: Baseline classifiers and data splitting.
- **classifier_evaluation.py**: Evaluation scripts for classifiers.


## Notes

- All user input is converted to lowercase for processing.
- Dialog acts are classified using either a baseline classifier or a ML classifier.
- The system uses Levenshtein distance for keyword matching.

## Configurability
These can be changed in the code file `state_transition.py` under the dictionary named `CONFIG`:
- Maximum correcting distance for Levenshtein edit distance 
- Always confirm for each preference or not 
- Output in all caps or not 
- Use baseline dialog act recognition 

## Group Number & Members
group number: F3
members:
- Majo Bednár
- Lonneke Hormann
- Finn Joosten
- Jop Paro