#
#
def filter_cond(line_dict):
    """Filter function
    Takes a dict with field names as argument
    Returns True if conditions are satisfied
    """
    if line_dict["if1"] == '':
        cond_match = False
    else:
        cond_match = (
        20 < float(line_dict["if1"]) < 40
        ) 
    return True if cond_match else False
