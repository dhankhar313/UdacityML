def outlierCleaner(predictions, features, labels):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    cleaned_data = []
    errors = abs(labels - predictions)
    errors = [item for sublist in errors for item in sublist]
    # errors = [round(i, 2) for i in errors]
    # print('Errors:', errors)
    features = [item for sublist in features for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    # labels = [round(i, 2) for i in labels]
    data = {}
    for i in range(len(errors)):
        data[errors[i]] = (features[i], labels[i], errors[i])
    errors = sorted(data.keys(), reverse=True)
    # print("Sorted errors:", errors)
    temp = int(len(features) / 10)
    errors = errors[temp:]
    # print('Shortened Errors:', errors)
    for i in errors:
        cleaned_data.append(data[i])
    # print(len(cleaned_data))
    # print(cleaned_data)
    return cleaned_data
