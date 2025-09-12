def prompt(classifier):
    while True:
        sentence = input('Type a sentence you want to classify a speech act of, or q() to finish\n')
        if sentence == 'q()':
            break
        print(classifier.predict_sentence(sentence))
