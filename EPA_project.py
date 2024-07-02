"""
Grace Li and Julie Henry
DS2500: Intermediate Programming with Data


Date: Tue Nov 28 13:44:07 2023

File: project.py
    
Description: Using EPA's Toxic Release Inventory data files from 2018-2022, 
analyze which state has the highest volume of chemical releases as well as 
the smallest volume. With the yearly data, perform a linear regression 
to test correlation between the year and the total volume of chemical releases.
Calculate various statistics for the user-inputted state and years. 

"""

# imports
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
import numpy as np
import re
from collections import defaultdict
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# constants
FILE_DIR = "chemical_data"

def get_filenames(dirname):
    """ Given a directory name, generate all the non-dir filenames
        at the top level and return them as a list of strings,
        including the directory as a prefix
        Parameters: dirname - name of the file directory
        Returns: list of strings
    """
    filelist = []
    files = os.listdir(dirname)
   
    for file in files:
        path = dirname + "/" + file
        if not os.path.isdir(path) and not file.startswith("."):
            filelist.append(path)
           
    return filelist

def read_csv(filename, skip_rows = 1):
    """ Given the name of a csv file, return its contents as a 2d list,
        including the header.
        Parameters: filename - string, skip_rows - int number of rows to skip
        Returns: 2d list
    """
    data = []
    
    with open(filename, "r") as infile:
        csvfile = csv.reader(infile)
        for _ in range(skip_rows):
            next(csvfile)
        for row in csvfile:
            data.append(row)
           
    return data

def extract_years(file_paths):
    """ Given the names of the files, match the varying names of the files
        with the corresponding years using the pattern of the years and
        appending it to the list in the original order.
        Parameters: file_paths - list of strings of the filenames
        Returns: list of years
    """
    years = []
    year_pattern = r"\d{4}"
   
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        year_match = re.search(year_pattern, filename)
       
        if year_match:
            years.append(year_match.group())
   
    return years

def extract_field_names(file_path):
    """ Given the names of one file, extract all the header names
        Parameters: file_path - name of a file
        Returns: names of the fields/headers
    """
    with open(file_path, mode= "r", newline= "", encoding = "utf-8") as file:
        csv_reader = csv.reader(file)
        field_names = next(csv_reader)

    return field_names

def convert_row_to_dict(row, field_names):
    """ Given one row of data and all the field names,
        converts the row to a dictionary
        Parameter: row - one row of data, field_names - names of all fields
        Returns: dictionary
    """
    return dict(zip(field_names, row))

def convert_3d_list_to_dicts(lst, file_paths, years):
    """ Given a 3d list, the name of the files, and the list of years,
        convert the 3d list to a dictionary of list of dictionaries where
        the key is the year and the value is a list of dictionaries per year
        Parameters: lst - 3d list, file_paths - name of files, years -
        list of years
        Returns: data_dict - dictionary of list of dictionaries
    """
    data_dict = {}
   
    for data_list, file_path, year in zip(lst, file_paths, years):
        field_names = extract_field_names(file_path)
       
        year_data = []
        for row in data_list:
            year_data.append(convert_row_to_dict(row, field_names))
       
        data_dict[year] = year_data
   
    return data_dict

def filter_columns(data, desired_columns):
    """ Given a dictionary of list of dictionaries, create a smaller
        dictionary that only has the columns we want to look at
        Parameters: data - dictionary of list of dictionaries,
        desired_columns - list of strings
        Returns: updated dictionary
    """
    filtered_data = {}
    
    for group_key, inner_list in data.items():
        filtered_inner_list = []
        for inner_dict in inner_list:
            filtered_inner_dict = {key: inner_dict[key] for key in
                                   desired_columns if key in inner_dict}
            filtered_inner_list.append(filtered_inner_dict)
        filtered_data[group_key] = filtered_inner_list
       
    return filtered_data

def float_convert(data_dict):
    """ Given one dictionary, convert all non-alphabetic characters to floats
        Parameters: data_dict - single dictionary
        Returns: None
    """
    for key, value in data_dict.items():
        if isinstance(value, str) and not value.isalpha():
            try:
                data_dict[key] = float(value)
            except ValueError:
                pass

def clean_non_alpha(dict_list):
    """ Given a dictionary of list of dictionaries, convert non-alphabetic
        characters to floats by iterating through the outer dictionary and
        into each individual row dictionary
        Parameters: dict_list - a dictionary of list of dictionaries
        Returns: None
    """
    for year, year_data in dict_list.items():
        for dct in year_data:
            float_convert(dct)
           
def convert_units_and_values(data):
    """ Given a dictionary of list of dictionaries, convert the total releases
        column when it gives grams to value in pounds to maintain consistency
        Parameters: data - a dictionary of list of dictionaries
        Returns: converted dictionary
    """
    for group_key, inner_list in data.items():
        for inner_dict in inner_list:
            if "47. UNIT OF MEASURE" in inner_dict and \
                inner_dict["47. UNIT OF MEASURE"] == "Grams":
                if "104. TOTAL RELEASES" in inner_dict:
                    grams_value = inner_dict["104. TOTAL RELEASES"]
                    pounds_value = grams_value * 0.00220462  
                    inner_dict["104. TOTAL RELEASES"] = pounds_value
                    inner_dict["47. UNIT OF MEASURE"] = "Pounds"

    return data

def calculate_mean_median(data, year, state):
    """ Given a dictionary of list of dictionaries, the user-given year and
        state, calculate the mean and median of the summed total releases per
        facility
        Parameters: data - a dictionary of list of dictionaries, year - string,
        state - string
        Returns: mean and median
    """
    facility_total_releases = defaultdict(float)

    for entry in data.get(year, []):
        if entry.get("8. ST") == state:
            facility_name = entry.get("4. FACILITY NAME")
            total_releases = entry.get("104. TOTAL RELEASES")
            if facility_name and total_releases is not None:
                facility_total_releases[facility_name] += total_releases

    summed_values = list(facility_total_releases.values())
    if summed_values:
        mean_volume = sum(summed_values) / len(summed_values)
        median_volume = sorted(summed_values)[len(summed_values) // 2] if \
            len(summed_values) % 2 != 0 else (summed_values[len(summed_values)\
            // 2 - 1] + summed_values[len(summed_values) // 2]) / 2
        return mean_volume, median_volume
    else:
        return None, None
   
def find_most_least_occurrences(data, year):
    """ Given a dictionary of list of dictionaries and the user-given year,
        find which facilities had the most and least number of chemical
        releases and which state they are located in
        Parameters: data 0 dictionary of list of dictionaries, year - string
        Returns: max_state name, max_facility name, min_state name,
        min_facility name
    """
    facility_occurrences = defaultdict(int)

    for state_data in data.values():
        for entry in state_data:
            facility_name = entry.get("4. FACILITY NAME")
            state = entry.get("8. ST")
            if facility_name and state:
                facility_occurrences[(state, facility_name)] += 1
               
    max_state_facility = max(facility_occurrences,
                             key=facility_occurrences.get)
    max_state, max_facility = max_state_facility

    min_state_facility = min(facility_occurrences,
                             key=facility_occurrences.get)
    min_state, min_facility = min_state_facility

    return max_state, max_facility, min_state, min_facility

def find_mode_facilities_per_state(data, year, state):
    """ Given a dictionary of list of dictionaries, a user-given year and
        state, find the facilities with the most rows/releases in that state
        that year
        Parameters: data - dictionary of list of dictionaries, year - string,
        state - string
        Returns: name of the facility
    """
    state_facility_count = defaultdict(int)

    for entry in data.get(year, []):
        if entry["8. ST"] == state:
            facility_name = entry["4. FACILITY NAME"]
            state_facility_count[facility_name] += 1

    max_count = max(state_facility_count.values())
    mode_facilities = [facility for facility, count in
                       state_facility_count.items() if count == max_count]

    return mode_facilities

def calculate_total_releases_and_years(data):
    """ Given a dictionary of list of dictionaries, calculate the total volume
        of chemical releases each year regardless of state
        Parameters: data - dictionary of list of dictionaries
        Returns: two lists
    """
    total_releases_by_year = []
    years_list = list(data.keys())

    for year, year_data in data.items():
        total_releases = sum(entry["104. TOTAL RELEASES"]
                             for entry in year_data)
        total_releases_by_year.append(total_releases)

    return total_releases_by_year, years_list

def linear_reg(lst, lst2, year_predict):
    """ Given two lists and the given year to predict, calculate the linear
        regression of the lists and use the calculated slope and intercept
        to predict a y-value for the given year
        Parameters: lst - list, lst2 - list, year_predict - string of year
        Returns: predict - predicted mean finish time time for given year
    """
    lr = stats.linregress(lst, lst2)
    predict = lr.slope * int(year_predict) + lr.intercept
   
    return predict

def plot_lr(x_lst, y_lst):
    """ Given two lists, plot a linear regression plot
        Parameters: x_lst - list for x-values, y_lst - list for y-values
        Returns: None
    """
    sns.regplot(x = x_lst, y = y_lst, color = "cadetblue")
   
    plt.xlabel("Year")
    plt.ylabel("Total Chemical Releases (in pounds)")
    plt.title("Linear Regression of Year vs. Total Chemical Releases")
    plt.xticks(range(int(min(x_lst)), int(max(x_lst)) + 1))
    plt.savefig("chemical_lr.pdf", bbox_inches = "tight")
    plt.show()
   
def calculate_total_releases_and_years_for_state(data, state_name):
    """ Given a dictionary of list of dictionaries and a state name,
        calculate the total releases per year
        Parameters: data - dictionary of list of dictionaries, state_name -
        string
        Returns: two lists
    """
    total_releases_for_state = {}
    years_set = set()
   
    for year, facilities in data.items():
        years_set.add(year)
        for facility in facilities:
            state = facility.get("8. ST")
            total_releases = float(facility.get("104. TOTAL RELEASES", 0))
           
            if state == state_name:
                if year not in total_releases_for_state:
                    total_releases_for_state[year] = 0.0
                total_releases_for_state[year] += total_releases
               
    total_releases_list = [total_releases_for_state.get(year, 0.0)
                           for year in sorted(list(years_set))]
   
    return total_releases_list, sorted(list(years_set))

def get_min_max_state_total_releases(data, states, year):
    """ Given a 2d list of state data and years and another list of state
        names in the same index order, return the minimum and maximum chemical
        release by volume states
        Parameters: data - 2d list, states - list, year - string of year
        Returns: min_state - string, max_state - string
    """
    year_index = None
    for idx, state_data in enumerate(data):
        if year in state_data[1]:
            year_index = state_data[1].index(year)
            break
   
    if year_index is None:
        return "Year not found in the data."
   
    total_releases_for_year = [state_data[0][year_index] for state_data in
                               data]
    min_state = states[total_releases_for_year.index(
        min(total_releases_for_year))]
    max_state = states[total_releases_for_year.index(
        max(total_releases_for_year))]
   
    return min_state, max_state

def corr_calc(lst, lst2):
    """Calculate the correlation between two lists
       Parameters: lst - list, lst2- list
       Returns: correlation coefficient
    """
    corr = np.corrcoef(lst, lst2)[0, 1]
   
    return corr

def state_corr(states_data, state_names):
    """ Given a nested list and a list of state name strings,
        create a dictionary by calling the corr_calc function as the value and
        using the name of the state as the key
        Parameters: states_data - 3d list, state_names - list
        Returns: dictionary
    """
    correlation_dict = {}

    for idx, state_data in enumerate(states_data):
        total_releases = state_data[0]
        years = state_data[1]
       
        total_releases = [float(value) for value in total_releases]
       
        years = [int(year) for year in years]
       
        corr = corr_calc(total_releases, years)
       
        correlation_dict[state_names[idx]] = corr

    return correlation_dict

def plot_states_lr(states_data, states_list):
    """ Given a 3d list and a list of state name strings,
        plot the linear regression of each state's total releases against
        years
        Parameters: states_data - 3d list, states_list - list
        Returns: None
    """
    plt.figure(figsize=(10, 6))
   
    for idx, state_data in enumerate(states_data):
        total_releases, years = state_data
        state_name = states_list[idx]
        years = list(map(int, years))
        sns.regplot(x=years, y=total_releases, scatter=True, label=state_name)
   
    plt.xlabel("Year")
    plt.ylabel("Total Releases")
    plt.title("Linear Regression of Year vs. Total Releases by State")
    plt.xticks(range(int(min(years)), int(max(years)) + 1))
    plt.legend()
    plt.savefig("states_lr.pdf", bbox_inches = "tight")
    plt.show()

def plot_bar(states_data, states_list):
    """ Given a 3d list and a list of state name strings, plot a bar chart
        that has 5 subbars per year for each of the 6 states
        Parameters: states_data - dictionary of list of dictionaries,
        states_list - list
        Returns: None
    """
    bar_width = 0.1
    n_states = len(states_data)
    x = range(n_states)
   
    colors = ["lightpink", "hotpink", "crimson", "magenta", "darkmagenta"]
   
    for i, state_data in enumerate(states_data):
        total_releases, years = state_data
        for j, year_total_release in enumerate(total_releases):
            plt.bar(
                i + (j - 2) * bar_width, year_total_release, width=bar_width,
                align="center", color=colors[j],
                label=years[j] if i == 0 else None)
   
    plt.xlabel("States")
    plt.ylabel("Total Releases")
    plt.title("Total Releases by State for Each Year")
    plt.xticks(x, states_list)
   
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = sorted(set(labels), key = labels.index)
    plt.legend(handles[:len(unique_labels)], unique_labels, title = "Years",
               bbox_to_anchor=(1.05, 1), loc="upper left")
   
    plt.tight_layout()
    plt.savefig("states_total_releases.pdf", bbox_inches = "tight")
    plt.show()
       
def extract_coordinates(data, target_year):
    """ Given a dictionary of a list of dictionaries and a year, extract
        latitude, longitude, and state data from the specified year's data.
        Parameters:
        - data: Dictionary of list of dictionaries.
        - target_year: The year for which data should be extracted.
        Returns:
        - latitudes: List of latitudes.
        - longitudes: List of longitudes.
        - states: List of state codes.
    """
    latitudes = []
    longitudes = []
    states = []

    state_to_number = {"MA": 0, "CT": 1, "RI": 2, "ME": 3, "NH": 4, "VT": 5}

    year_data = data.get(target_year, [])

    for entry in year_data:
        latitude = entry.get("12. LATITUDE")
        longitude = entry.get("13. LONGITUDE")
        state = entry.get("8. ST")
        facility_name = entry.get("4. FACILITY NAME")

        if latitude is not None and longitude is not None and state is not \
            None:
            try:
                latitudes.append(float(latitude))
                longitudes.append(float(longitude))
                states.append(state_to_number.get(state, -1))
            except ValueError:
                print(f"No location data for {facility_name} in {state}")

    return latitudes, longitudes, states

def plot_coordinates(latitudes, longitudes, states, target_year):
    """ Given latitude, longitude, and state data, plot a scatter plot
        color-coded by state.
        Parameters:
        - latitudes: List of latitudes.
        - longitudes: List of longitudes.
        - states: List of state codes.
        - target_year: The year for which the data is plotted.
        Returns:
        - None
    """
    state_to_number = {"MA": 0, "CT": 1, "RI": 2, "ME": 3, "NH": 4, "VT": 5}
    number_to_state = {v: k for k, v in state_to_number.items()}
   
    if latitudes and longitudes and states:
        plt.scatter(longitudes, latitudes, c=states, cmap="plasma", marker="o")
        plt.title(f"Facility Locations by State in {target_year}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        unique_states = sorted(set(states))
        norm = Normalize(vmin=min(unique_states), vmax=max(unique_states))
        sm = cm.ScalarMappable(cmap='plasma', norm=norm)
        sm.set_array([])
        legend_labels = [f"{number_to_state[state]}: {state}" for state in \
                         unique_states]
        plt.legend(title='States', handles=[plt.Line2D([0], [0], marker='o', \
                color='w', label=label, markerfacecolor=sm.to_rgba(state))
                for state, label in zip(unique_states, legend_labels)],
                   loc='upper left', bbox_to_anchor=(1.2, 1.0))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"facility_locations_by_state_{target_year}.pdf", \
                    bbox_inches = "tight")
        plt.show()

def main():
   
    # create list of names of all files in directory
    files_lst = get_filenames(FILE_DIR)
   
    # read the information in each file into a 3d list
    lst = [read_csv(file) for file in files_lst]

    # match filenames with the correct corresponding year
    years = extract_years(files_lst)
   
    # convert 3d list into a dict of list of dicts where keys are years
    dct = convert_3d_list_to_dicts(lst, files_lst, years)
   
    desired_columns = ["4. FACILITY NAME", "8. ST", "12. LATITUDE",
            "13. LONGITUDE", "47. UNIT OF MEASURE","104. TOTAL RELEASES"]
   
    # create smaller dict with only keys of the desired columns above
    small_dct = filter_columns(dct, desired_columns)
   
    # convert numeric datatypes to floats and convert grams to pounds
    clean_non_alpha(small_dct)
    updated_data = convert_units_and_values(small_dct)
 
    # ask for user input on what year and state to analyze
    user_year = input(f"\nWhat year do you want to calculate chemical release "
                      f"statistics for? ")
    user_state = input(f"\nWhat state do you want to calculate chemical "
                       f"release statistics for? ")

    # statistical analysis computations
    mean, median = calculate_mean_median(small_dct, user_year,
                                                user_state)
   
    max_state, max_facility, min_state, min_facility \
        = find_most_least_occurrences(small_dct, user_year)
       
    mode_facilities_for_state = find_mode_facilities_per_state(small_dct,
                                                user_year, user_state)
   
    total_releases, years = calculate_total_releases_and_years(small_dct)
    years_as_ints = [int(year) for year in years]
   
    prediction_2023 = linear_reg(years_as_ints, total_releases, "2023")
    pred_rounded = int(prediction_2023)
   
    plot_lr(years_as_ints, total_releases)
   
    # gather lists of years and total releases for each state
    states_data = []
    states = ["MA", "CT", "ME", "NH", "RI", "VT"]
    for state in states:
        total_releases, years = calculate_total_releases_and_years_for_state(
            small_dct, state)
        states_data.append([total_releases, years])
       
    # min and max total release state from user year
    min_state, max_state = get_min_max_state_total_releases(states_data,
                                                            states,
                                                            user_year)

    # calculate the correlation coefficients for each state
    corr_dict = state_corr(states_data, states)
   
    # plot the linear regression lines for each state
    plot_states_lr(states_data, states)
   
    # plot bargraph
    plot_bar(states_data, states)
   
    # plot latitude and longitude of facilties
    latitudes, longitudes, states = extract_coordinates(small_dct, user_year)
    plot_coordinates(latitudes, longitudes, states, user_year)

    # communicate
    print(f"\nStatistics:")
    print(f"\nThe mean volume of chemical releases in {user_state} in "
          f"{user_year} is {round(mean,2)} lbs.")
    print(f"\nThe median volume of chemical releases in {user_state} in "
          f"{user_year} is {round(median,2)} lbs.")
    print(f"\nIn {user_year}, the facility with the most occurences of "
          f"chemical releases is {max_facility} in {max_state}.")
    print(f"\nIn {user_year}, the facility with the least occurences of "
          f"chemical releases is {min_facility} in {min_state}.")
    print(f"\nThe facility or facilities with the most rows/releases in "
          f"{user_state} for {user_year} is {mode_facilities_for_state[0]}.")
    print(f"\nThe state with the largest volume of chemical releases in "
          f"{user_year} is {max_state}.")
    print(f"\nThe state with the smallest volume of chemical releases in "
          f"{user_year} is {min_state}.")
    print(f"\nLinear Regression:")
    print(f"\nCorrelation coefficients between year and total releases in "
          f"each state: {corr_dict}")
    print(f"\n{pred_rounded} pounds of chemicals are predicted to be "
          f"released in 2023.")


if __name__ == "__main__":
    main()