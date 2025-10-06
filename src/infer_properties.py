import pandas as pd


class InferredProperties:
    """Class for inferring additional properties about a restaurant."""
    def __init__(self, restaurant_name=None, df=None):
        self.touristic = None
        self.assigned_seats = None
        self.children = None
        self.romantic = None
        if restaurant_name and df is not None:
            self.infer_properties(restaurant_name, df)

    def infer_properties(self, restaurant_name, df):
        """Infers additional properties about a restaurant."""
        restaurant = df.loc[df['restaurantname'] == restaurant_name].squeeze()
        if restaurant['pricerange'] == 'cheap' and restaurant['foodquality'] == 'good':
            self.touristic = True
        if restaurant['food'] == 'romanian':
            self.touristic = False
        if restaurant['lengthofstay'] == 'long stay':
            self.children = False
            self.romantic = True
        if restaurant['crowdedness'] == 'busy':
            self.assigned_seats = True
            self.romantic = False

    def is_preference_satisfied(self, preference):
        """Returns True if a preference is satisfied."""
        if preference is None:
            return True
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
        """Neatly prints inferred properties."""
        return (
            f'Inferred properties:\n'
            f'touristic:      {self.touristic}\n'
            f'assigned seats: {self.assigned_seats}\n'
            f'children:       {self.children}\n'
            f'romantic:       {self.romantic}'
        )


def preference_reasoning(preference: str):
    """Gives a reason why a preference about a restaurant is satisfied."""
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
    if preference is None:
        return ''


def main():
    """Testing the inference of properties."""
    restaurant_df = pd.read_csv('data/restaurant_info.csv')
    restaurant = 'thanh binh'
    print(InferredProperties(restaurant, restaurant_df))


if __name__ == '__main__':
    main()
