


def separate_integers(string):
    integers = []
    current_number = ""
    for char in string:
        if char.isdigit():
            current_number += char
        else:
            if current_number:
                integers.append(int(current_number))
                current_number = ""
    if current_number:
        integers.append(int(current_number))
    return integers