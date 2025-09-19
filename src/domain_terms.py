from data_management import load_df_from_csv

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


def main():
    df = load_df_from_csv('data/restaurant_info.csv')
    print(df['pricerange'].unique())
    print(df['area'].unique())
    print(df['food'].unique())


if __name__ == '__main__':
    main()
