import pandas as pd 


"""
Function that takes as input pricerange area and food and recommends a restaurant.
The lookup function prioritices the restaurant which fits on the most criterea.
If there are multiple restaurants with the same number of corresponding criterea the importance wil be as follows:
1. pricerange 
2. food 
3. area
If there are restaurants with the same ammount of fittness a list is returned
"""
def restaurant_lookup(df, pricerange, area, food):
    # Loop through all restaurants and save a list with scores: 
    # If restaurant has good price range add 1.1 
    # If restaurant has good area add 1
    # If restaurant has good food add 1.05
    # Return restaurant(s) with hihgest score

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

    # Filter out restaurants with score > 3
    suitable_restaurants = df.loc[df['score'] > 3, 'restaurantname'].tolist()

    return suitable_restaurants


def main():
    restaurant_info = pd.read_csv('data/restaurant_info.csv')
    print(restaurant_info)
    print(restaurant_lookup(restaurant_info, 'moderate', 'centre', 'italian'))
if __name__ == "__main__":
    main()
