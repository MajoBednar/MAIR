import pandas as pd 


def restaurant_lookup(df, pricerange, area, food, priority=False):
    """
    Function that takes as input pricerange, area and food and recommends a restaurant.
    The lookup function can prioritize the restaurant which fits on the most criteria (achieves highest score).
    The criteria of importance are as follows:
    1. pricerange
    2. food
    3. area
    If there are restaurants with the same amount of fittness a list is returned.
    """
    scores = []
    for _, row in df.iterrows():
        score = 0
        if row['pricerange'] == pricerange or pricerange == 'any':
            score += 1.1
        if row['area'] == area or area == 'any':
            score += 1
        if row['food'] == food or food == 'any':
            score += 1.05
        scores.append(score)

    # Add scores to the dataframe
    df['score'] = scores

    # Filter out restaurants with score > 3, that is, all preferences are satisfied
    if priority is False:
        return df.loc[df['score'] > 3, 'restaurantname'].tolist()

    # If you want to prioritize based on criteria, get the maximum score return those restaurants
    max_score = df['score'].max()
    return df.loc[df['score'] == max_score, 'restaurantname'].tolist()


def main():
    """Testing restaurant lookup."""
    restaurant_info = pd.read_csv('data/restaurant_info.csv')
    print(restaurant_info)
    print(restaurant_lookup(restaurant_info, 'moderate', 'centre', 'italian'))


if __name__ == "__main__":
    main()
