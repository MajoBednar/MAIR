import pandas as pd


class InferredProperties:
    def __init__(self):
        self.touristic = None
        self.assigned_seats = None
        self.children = None
        self.romantic = None

    def is_preference_satisfied(self, preference):
        if preference == 'touristic' and self.touristic is True:
            return True
        if preference == 'not touristic' and self.touristic is False:
            return True
        if preference == 'assigned seats' and self.assigned_seats is True:
            return True
        if preference == 'not children' and self.children is False:
            return True
        if preference == 'romantic' and self.romantic is True:
            return True
        if preference == 'not romantic' and self.romantic is False:
            return True
        return False

    def __str__(self):
        return (
            f'Inferred properties:\n'
            f'touristic:      {self.touristic}\n'
            f'assigned seats: {self.assigned_seats}\n'
            f'children:       {self.children}\n'
            f'romantic:       {self.romantic}'
        )


def infer_properties(restaurant_name, df):
    print(restaurant_name)
    restaurant = df.loc[df['restaurantname'] == restaurant_name].squeeze()
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


def preference_reasoning(preference: str):
    main_clause = f'. The restaurant is {preference} because '
    if preference == 'touristic':
        return main_clause + 'it has cheap and good food'
    if preference == 'not touristic':
        return main_clause + 'romanian food is unknown to tourists'
    if preference == 'assigned seats':
        return f'. The restaurant has {preference} because it is busy'
    if preference == 'not children':
        return f'. The restaurant is for a long stay which is not suitable for children'
    if preference == 'romantic':
        return main_clause + 'it allows you to stay for a long time'
    if preference == 'not romantic':
        return main_clause + 'it is busy'


def main():
    restaurant_df = pd.read_csv('data/restaurant_info.csv')
    restaurant = 'thanh binh'
    print(infer_properties(restaurant, restaurant_df))


if __name__ == '__main__':
    main()
