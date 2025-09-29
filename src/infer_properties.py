import pandas as pd


class InferredProperties:
    def __init__(self):
        self.touristic = None
        self.assigned_seats = None
        self.children = None
        self.romantic = None

    def __str__(self):
        return (
            f'Inferred properties:\n'
            f'touristic:      {self.touristic}\n'
            f'assigned seats: {self.assigned_seats}\n'
            f'children:       {self.children}\n'
            f'romantic:       {self.romantic}'
        )


def infer_properties(restaurant):
    print(restaurant)
    inferred_properties = InferredProperties()
    if restaurant['pricerange'] == 'cheap' and restaurant['foodquality'] == 'good':
        inferred_properties.touristic = True
    if restaurant['food'] == 'romanian':
        inferred_properties.touristic = False
    if restaurant['lengthofstay'] == 'long stay':
        inferred_properties.children = False
        inferred_properties.romantic = True
    if restaurant['crowdedness'] == 'busy':
        inferred_properties.assigned_seats = True
        inferred_properties.romantic = False
    return inferred_properties


def main():
    restaurant_df = pd.read_csv('data/restaurant_info.csv')
    restaurant = restaurant_df.iloc[14]
    print(infer_properties(restaurant))


if __name__ == '__main__':
    main()
