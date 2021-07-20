# Team Matthias & Boris: ane_energy imbalance pricing


# Main data packages. 
import numpy as np
import pandas as pd

# function that gives the columns in the "EQ_epex_da_prognosis.csv" file names
def eq_epex_da_prognosis_df_col_names(df):
    df.columns = ["date_prognosticated", "prognosis", "date_issued"]
    return df


def prognosis_count(df):
    """
    takes a given DataFrame as parameter and creates an extra column
    "prognosis_count", where the numbers of
    prognoses per unique date are counted
    """
    # Add empty column to eq_epex_da_prognosis_df called "prognosis_count" 
    # to count the numbers of prognoses for a given time stamp:
    df["prognosis_count"] = np.nan
    # In a for loop, the amounts of prognoses per unique date are counted
    # and saved in th prognosis_count column
    j = 1
    df["prognosis_count"][0]= int(1)
    for i in range (1, len(df["date_prognosticated"])):
        if df["date_prognosticated"][i] == df["date_prognosticated"][i-1]:
            j += 1
            df["prognosis_count"][i] = int(j)
        else:
            j = 1
            df["prognosis_count"][i] = int(j)
    # transforms the values in prognosis_count to integers
    df["prognosis_count"] = df["prognosis_count"].values.astype(int)
    # resets the index
    df.reset_index()
    # saves the df to a csv file in the data/data_processed folder 
    df.to_csv("data/data_processed/EQ_epex_da_prognosis_count.csv", index=False)
    
    return df
    

    
def current_prognosis(Input_df):
    """
    This function takes the "eq_epex_da_prognosis_count_df"  from the folder 
    data/data_processed)as parameter "Input_df"
    and gets rid of all rows with older prognoses
    """
    # Create a dictionary of nested dictionaries which contain the number of 
    # prognosis_counts as keys and the prognoses as value for each unique 
    # date_prognosticated as key:
    dict = Input_df.groupby("date_prognosticated")[["prognosis_count", "prognosis"]] \
      .apply(lambda x: x.set_index("prognosis_count").to_dict(orient="index")) \
      .to_dict()

    # initialize empty Data Frame and fill it with input from nested dict:
    eq_epex_da_last_prognosis_df = pd.DataFrame(columns=["timestamp", "prognosis"])
    
    # iterate through the nested dict keys and append the "timestamp" as key and
    # the last_prognosis as "prognosis":
    for key in dict:
        last_prognosis = list(dict[key].values())[-1]
        eq_epex_da_last_prognosis_df = eq_epex_da_last_prognosis_df.append({'timestamp': key,'prognosis':last_prognosis["prognosis"],
                                                                           }, ignore_index=True)
    # save the resulting DataFrame into a csv file:
    eq_epex_da_last_prognosis_df.to_csv("data/data_processed/EQ_epex_da_last_prognosis_count.csv", index=False)
    
    return eq_epex_da_last_prognosis_df


# takes the string of the data file path for one of the "regelleistung.csv" files  and turns it into a Data Frame that serves as input for the
# regelleistung_transform() function:


def regelleistung_df(file_path):
    # Import data
    regelleistung_aggr_results_df = pd.read_csv(file_path)
    # Get rid of the date_end column, as it contains the same values like the date_start column:
    regelleistung_aggr_results_df.drop(["date_end"], axis=1, inplace=True)
    # Add a column to the Data Frame "regelleistung_aggr_results_df" which is the combination of the columns "product" and "reserve_type" 
    # and  the column shall be named "product_reserve_type". This will be our unique key.
    regelleistung_aggr_results_df.eval("product_reserve_type= product + reserve_type", inplace=True)
    # drop unnecessary columns:
    regelleistung_aggr_results_df.drop(["product", "reserve_type"], axis=1, inplace=True)
    return regelleistung_aggr_results_df



def regelleistung_transform(regelleistung_df, file_name):
    """
    This function takes in a modified (see above) "regelleistung_df" plus a string-name for the
    csv-datfile which saves the end result as input parameters
    and writes all multiple rows who have the same time stamp into a 
    new DataFrame stored as the new csv file), where
    all those row cells are written into newly named columns in one row per unique time stamp
    """
    # build a list of the unique entries in the "product_reserve_type" column:
    product_reserve_type_list = regelleistung_df["product_reserve_type"].unique().tolist()
    #print(product_reserve_type_list)
    # create a list of all column names from the csv file "regelleistung_aggr_results.csv":
    unique_columns = regelleistung_df.columns.tolist()
    
    # take all the unique time stamps form "regelleistung_aggr_results.csv" and create a list:
    unique_dates = regelleistung_df["date_start"].unique().tolist()
    
    # construct a (mostly empty) array(df) from the unique date_starts and with columns with a prefix from the
    # column "product_reserve_type" plus the unique_columns from the basic list(columns 1 to 13),
    # then fill the cells of the new columns with 0s
    
    repeat_col_number = regelleistung_df.shape[1] - 2
    
    df_test = pd.DataFrame(unique_dates,columns=["date_start"])  
    for product_reserve_type in product_reserve_type_list:
        for column in unique_columns[1:repeat_col_number+1]:
            df_test[f"{product_reserve_type}_{column}"] = 0.00
    print(df_test.shape, len(regelleistung_df["product_reserve_type"]) - 1)
    
    # create a list of all column names from "df_test":
    df_test_columns = df_test.columns.unique().values.tolist()
    
    # take the information from the cells from the additional same timestamp rows and
    # write it into the corresponding cells of "df_test" to transform the multiline
    # data content of the original csv into one row in the dataframe "df_test":

    row_enum_input, row_enum_output, prt_list_enum, m = 0, 0, 0, 0
    col_enum_input, col_enum_output = 1, 1
    
    # calculate parameters which are specific to the input_df to improve reusability of function:
    
    max_row_input_df = len(regelleistung_df["product_reserve_type"]) - 1
    max_row_output_df = len(df_test["date_start"]) - 1
    max_col_enum_output_df = df_test.shape[1] - 1
    
    max_repeats = len(product_reserve_type_list) - 1
    print(f"max_row_input_df:{max_row_input_df}\nmax_row_output_df:{max_row_output_df}\nmax_col_enum_output_df:{max_col_enum_output_df}")
    print(f"repeat_col_number: {repeat_col_number}\nmax_repeats: {max_repeats}")
    
    # below is the code for filling the "test_df" template csv with data from the input_df:
    
    for row_enum_input in range(max_row_input_df + 1):
        col_sec_it = 1
        l = 0
        while col_enum_output <= max_col_enum_output_df + 1:
            if row_enum_input == max_row_input_df or row_enum_output > max_row_output_df :
                break
            elif regelleistung_df["product_reserve_type"][row_enum_input] == product_reserve_type_list[prt_list_enum] and regelleistung_df["date_start"][row_enum_input] == df_test["date_start"][row_enum_output] and col_enum_output < max_col_enum_output_df:
                
                for l in range (col_sec_it, col_sec_it + repeat_col_number):  
                    df_test[df_test_columns[l]][row_enum_output] = regelleistung_df[unique_columns[col_enum_input]][row_enum_input]
                    col_enum_input += 1
                    col_enum_output += 1
                if m < max_repeats:
                    m += 1
                    prt_list_enum += 1 
                    col_enum_input = 1
                    col_sec_it = 1 + m * repeat_col_number
                    if row_enum_input < len(regelleistung_df["product_reserve_type"]) - 1:
                        row_enum_input += 1
                
            elif regelleistung_df["product_reserve_type"][row_enum_input] != product_reserve_type_list[prt_list_enum] and regelleistung_df["date_start"][row_enum_input] == df_test["date_start"][row_enum_output] and col_enum_output < max_col_enum_output_df:
                if m < max_repeats:
                    col_enum_output += repeat_col_number
                    prt_list_enum += 1
                    m += 1
                    col_sec_it = 1 + m * repeat_col_number
                col_enum_input = 1
                
            else:
                col_sec_it = 1
                col_enum_output = 1
                col_enum_input = 1
                prt_list_enum = 0
                row_enum_input += 1
                row_enum_output += 1
                m = 0
                l = 0             
                           
    print(prt_list_enum, col_sec_it, row_enum_input, row_enum_output) 
    df_test.to_csv(f'data/data_processed/{file_name}.csv', index=False)
    
    return df_test.head(5)
    