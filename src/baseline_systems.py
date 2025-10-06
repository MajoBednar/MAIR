import re


class AlwaysLabelInform:
    """
    A baseline system that, regardless of the content of the utterance,
    always assigns the majority class of in the data. --> Majority class is inform.
    """
    def predict(self, features):
        """Predict a list of labels from a list of inputs."""
        return ['inform'] * len(features)

    def predict_sentence(self, utterance):
        """Predicts a dialog act of a single raw sentence."""
        return self.predict([utterance])


class BaselineRules:
    """Class that implements baseline rules found from empirical evaluation of the dataset."""
    def predict(self, features):
        """Predict a list of labels from a list of inputs."""
        predictions = []
        for utterance in features:
            # Set standard label to inform
            prediction = 'inform'
            # If yes is in utterance label affirm
            prediction = 'affirm' if bool(re.search('yes', utterance, flags=re.IGNORECASE)) else prediction
            # If goodbye or bye is in utterance label bye
            prediction = 'bye' if bool(re.search('bye|goodbye', utterance, flags=re.IGNORECASE)) else prediction
            # If the word no is in utterance label negate
            prediction = 'negate' if bool(re.search(r'\bno\b', utterance, flags=re.IGNORECASE)) else prediction
            # If thank is in utterance label thankyou
            prediction = 'thankyou' if bool(re.search('thank', utterance, flags=re.IGNORECASE)) else prediction
            # If cough, unintelligible, noise or sil is in utterance label null
            prediction = 'null' if bool(
                re.search('cough|unintelligible|noise|sil', utterance, flags=re.IGNORECASE)
            ) else prediction
            # If what, phone number, address, post code or postcode is in utterance label request
            prediction = 'request' if bool(
                re.search('phone number|address|post code|what|postcode', utterance, flags=re.IGNORECASE)
            ) else prediction
            # If another is in utterance label requalts
            prediction = 'requalts' if bool(re.search('another', utterance, flags=re.IGNORECASE)) else prediction

            predictions.append(prediction)
        return predictions

    def predict_sentence(self, utterance):
        """Predicts a dialog act of a single raw sentence."""
        return self.predict([utterance])


def main():
    """Testing baseline systems."""
    print(BaselineRules().predict_sentence('goodbye'))


if __name__ == "__main__":
    main()
