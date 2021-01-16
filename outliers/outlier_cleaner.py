def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    cleaned_data = []
    errors = abs(net_worths - predictions)
    errors = [item for sublist in errors for item in sublist]
    # errors = [round(i, 2) for i in errors]
    # print('Errors:', errors)
    ages = [item for sublist in ages for item in sublist]
    net_worths = [item for sublist in net_worths for item in sublist]
    # net_worths = [round(i, 2) for i in net_worths]
    data = {}
    for i in range(len(errors)):
        data[errors[i]] = (ages[i], net_worths[i], errors[i])
    errors = sorted(data.keys(), reverse=True)
    # print("Sorted errors:", errors)
    temp = int(len(ages) / 10)
    errors = errors[temp:]
    # print('Shortened Errors:', errors)
    for i in errors:
        cleaned_data.append(data[i])
    # print(len(cleaned_data))
    # print(cleaned_data)
    return cleaned_data
