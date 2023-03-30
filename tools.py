def str_to_int_list(values):
    try:
        og_values = values  # to track errors
        values = ''.join(values.split())
        splits = values.split(sep=",")
        return remove_duplicates([int(s) for s in splits])
    except ValueError:
        print("Reading config unsuccessful, check the formatting of: " + og_values)
        exit(0)


def str_to_str_list(values):
    try:
        og_values = values  # to track errors
        values = ''.join(values.split())
        return remove_duplicates(values.split(sep=","))
    except ValueError:
        print("Reading config unsuccessful, check the formatting of: " + og_values)
        exit(0)


def remove_duplicates(input_list):
    return list(dict.fromkeys(input_list))
