def qFirst4(trainSet):
    leanings = trainSet
    for leaning in leanings:
        for tree in leaning:
            for word in tree.get_words().split(" "):
                word = word.replace("\n", "")
                wordDict[word] = wordDict.get(word, len(wordDict))


def qSecond4(trainSet):
    """
    You'll notice that actual words didn't appear in the last question.
    Array indices are nicer to work with than words, so we typically
    write a dictionary encoding the words as numbers. This turns
    review strings into lists of integers. You will count the occurrences
    of each integer in reviews of each class.
    The infile has one review per line, starting with the rating and then a space.
    Note that the "words" include things like punctuation and numbers. Don't worry
    about this distinction for now; any string that occurs between spaces is a word.
    You must do three things in this question: build the dictionary,
    count the occurrences of each word in each rating and count the number
    of reviews with each rating.
    The words should be numbered sequentially in the order they first appear.
    counts[ranking][word] is the number of times word appears in any of the
    reviews corresponding to ranking
    nrated[ranking] is the total number of reviews with each ranking
    """
#     leanings = trainSet
#     for leaning in leanings:
#         for tree in leaning:
#             for word in tree.get_words().split(" "):
#                 word = word.replace("\n", "")
#                 wordDict[word] = wordDict.get(word, len(wordDict))

#     nrated = [0] * 3
    leanings = trainSet

    for i, leaning in enumerate(leanings):
        for tree in leaning:
            nrated[i] += 1
            for word in tree.get_words().split(" "):
                word = word.replace("\n", "")
                counts[i][wordDict[word]] += 1
            
    

def q5(alpha=1):
    """
    Now you'll fit the model. For historical reasons, we'll call it F.
    F[rating][word] is -log(p(word|rating)).
    The ratings run from 0-4 to match array indexing.
    Alpha is the per-word "strength" of the prior (as in q2).
    (What might "fairness" mean here?)
    """


#     F = [ [0] * len(wordDict) for _ in range(3)]
    for ratingIndex, ratingCount in enumerate(counts):
        summedCount = sum(ratingCount) + (alpha * len(wordDict))
        for wordIndex, wordCount in enumerate(ratingCount):
            prob = (wordCount + alpha) / float(summedCount)
            F[ratingIndex][wordIndex] = -log(prob+0.0000000000001)


def q6(testSet):
    """
    Test time! The infile has the same format as it did before. For each review,
    predict the rating. Ignore words that don't appear in your dictionary.
    Are there any factors that won't affect your prediction?
    You'll report both the list of predicted ratings in order and the accuracy.
    """

    
    predictions = []
    correct = 0
    count = 0
    for leaningIndex, leaning in enumerate(testSet):
        for review in leaning:
            priorList = [-log(x / float(sum(nrated))) for x in nrated]
            for word in review.get_words().split(" "):
                if word in wordDict:
                    for ratingIndex in range(3):
                        priorList[ratingIndex] += F[ratingIndex][wordDict[word]]
            bestPrediction = 0
            minVal = float("inf")
            for i in range(len(priorList)):
                if priorList[i] <= minVal:
                    minVal = priorList[i]
                    bestPrediction = i
            count += 1
#             print bestPrediction, leaningIndex
            if bestPrediction == leaningIndex:
                correct += 1
            predictions.append(bestPrediction)
        
    return (predictions, correct / float(count))


def q7():
    """
    Alpha (q5) is a hyperparameter of this model - a tunable option that affects
    the values that appear in F. Let's tune it!
    We've split the dataset into 3 parts: the training set you use to fit the model
    the validation and test sets you use to evaluate the model. The training set
    is used to optimize the regular parameters, and the validation set is used to
    optimize the hyperparameters. (Why don't you want to set the hyperparameters
    using the test set accuracy?)
    Find and return a good value of alpha (hint: you will want to call q5 and q6).
    What happens when alpha = 0?
    """

    bestAlpha = 0
    bestAccuracy = 0
    for alpha in [x * 0.1 for x in range(0, 20)]:
        q5(alpha)
        _, accuracy = q6(infile)
        if accuracy >= bestAccuracy:
            bestAlpha = alpha
            bestAccuracy = accuracy
    return bestAlpha

def q8():
    """
    We can also "hallucinate" reviews for each rating. They won't make sense
    without a language model (for which you'll have to take CS287), but we can
    list the 3 most representative words for each class. Representative here
    means that the marginal information it provides (the minimal difference between
    F[rating][word] and F[rating'][word] across all rating' != rating) is maximal.
    You'll return the strings rather than the indices, and in decreasing order of
    representativeness.
    """
    representatives = []
    for rating in range(3):
        wordList = []
        for word in wordDict:
            maxDiff = -111000
            for rPrime in range(3):
                if rating != rPrime:
                    diff = F[rating][wordDict[word]] - F[rPrime][wordDict[word]]
                    maxDiff = max(maxDiff, diff)
            wordList.append((maxDiff, word))
        sortedLst = sorted(wordList, key = lambda x: x[0])
        representatives.append([sortedLst[i][1] for i in range(3)])
    return representatives