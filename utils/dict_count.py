import jraph
def hashable_value(value):
    """
    Recursively convert a dictionary to a hashable type.
    """
    if isinstance(value, dict):
        return tuple((key, hashable_value(inner_value)) for key, inner_value in sorted(value.items()))
    elif isinstance(value, list):
        return tuple((f"layer_{i}", hashable_value(item)) for i, item in enumerate(value))
    elif isinstance(value, jraph.GraphsTuple):
        return ("GraphsTuple", tuple(hashable_value(item) for item in value))
    else:
        return value


def count_same_dicts(dict_list):
    count_dict = {}

    for d in dict_list:
        # Convert the dictionary to a hashable type
        hashable_dict_value = hashable_value(d)

        # Update the count
        count_dict[hashable_dict_value] = count_dict.get(hashable_dict_value, 0) + 1

    # Filter out dictionaries that occur only once
    same_dicts_count = {k: v for k, v in count_dict.items()}

    return same_dicts_count

def compare_and_replace(str1, str2):
    """
    Compare two strings element by element and replace equal elements with 'X'.
    """
    result = ''
    for char1, char2 in zip(str1, str2):
        if char1 == char2:
            result += '_'
        else:
            result += char1
    return result


if(__name__ == "__main__"):
    # Example usage:
    list_of_dicts = [
        {'a': 1, 'b': {'c': 2}},
        {'b': {'c': 2}, 'a': 1},
        {'c': 3, 'd': 4},
        {'a': 1, 'b': {'c': 2}},
        {'e': 5, 'f': 6},
        {'c': 3, 'd': 4},
    ]

    result = count_same_dicts(list_of_dicts)
    print(result)