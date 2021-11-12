import pandas as pd
import numpy as np
import os


class DataLoader:
    """
    This class works as abstraction layer for data acquisition
    """

    def __init__(self, p_time_period=pd.date_range('2014-01-24', '2017-10-24', freq='D').to_period(),
                 p_target=['Close'],
                 p_data_directory_general_internal='data/internal/',
                 p_data_directory_general_external='data/external/',
                 p_data_subdirectory_stocks='price-volume-data-for-all-us-stocks-etfs/Data/Stocks/',
                 p_data_subdirectory_etf='price-volume-data-for-all-us-stocks-etfs/Data/ETFs/',
                 p_data_subdirectory_ggl_trnds='googletrends/',
                 p_data_subdirectory_wkpd='wikipedia/',
                 p_every_xth_data=45,
                 p_rename_columns=True,
                 p_rename_target=True,
                 p_do_load_internal_stocks=True,
                 p_do_load_internal_etfs=False,
                 p_do_load_external_ggl_trnds=True,
                 p_do_load_external_wkpd=True,
                 p_replace_nan_with_mean=True,
                 p_select_only_few_targets=True  # maybe find a better parameter name for this...
                 ):
        self.time_period = p_time_period
        self.target = p_target  # Note the warning at parameter rename_target, some lines below

        self.data_directory_general_internal = p_data_directory_general_internal
        self.data_directory_general_external = p_data_directory_general_external

        self.data_subdirectory_stocks = p_data_subdirectory_stocks
        self.data_subdirectory_etf = p_data_subdirectory_etf
        self.data_subdirectory_ggl_trnds = p_data_subdirectory_ggl_trnds
        self.data_subdirectory_wkpd = p_data_subdirectory_wkpd

        self.every_xth_data = p_every_xth_data  # for internal data only, todo :: do this more specific, and use percentage

        self.rename_columns = p_rename_columns  # rename columns to include e.g. filename, for uniqueness. Has to be true as of now
        self.rename_target = p_rename_target  # rename the target column to be "Target", in addition to param 'rename_columns'
        # WARNING! Setting above paramter to True requires to have exactly one Target-Feature

        self.do_load_internal_stocks = p_do_load_internal_stocks
        self.do_load_internal_etfs = p_do_load_internal_etfs
        self.do_load_external_ggl_trnds = p_do_load_external_ggl_trnds
        self.do_load_external_wkpd = p_do_load_external_wkpd

        self.replace_nan_with_mean = p_replace_nan_with_mean

        self.select_only_few_for_targets = p_select_only_few_targets

    def get_target_columns(self):
        """
        GET-Function to retrieve the list of current targets
        :return: a list with the names of the targets
        """
        return self.target

    def set_target_columns(self, p_new_targets: [str]):
        """
        SET-Function to set the list of current targets
        :param p_new_targets:
        """
        self.target = p_new_targets

    def generate_complete_dataframe(self):
        """
        Generates the complete dataframe that contains all the available internal and external data
        Returns two dataframes, each indexed by date:
        - First: Dataframe that contains the information, excluding the target columns
        - And: Dataframe that contains the target values, shifted by one day (the values that are to be predicted)
        :return: Two dataframes indexed by date. First contains the values. Second contains the targets.
        """
        tmp_df_dict_values = {}  # holds the intermediate dataframes before the merge
        tmp_df_dict_targets = {}  # holds the intermediate target dataframe values before the merge

        if self.do_load_external_ggl_trnds:
            tmp_df_dict_values['ext_ggl'] = self.load_external_ggl_trnds_data()

        if self.do_load_external_wkpd:
            tmp_df_dict_values['ext_wkpd'] = self.load_external_wkpd_data()

        if self.do_load_internal_etfs:
            tmp_df_dict_values['int_etfs'], tmp_df_dict_targets['int_etfs'] = self.load_internal_data(False)

        if self.do_load_internal_stocks:
            tmp_df_dict_values['int_stocks'], tmp_df_dict_targets['int_stocks'] = self.load_internal_data(True)

        # print("Done reading all csv files seperately... creating a large combined df now")
        res_df_values = pd.concat(tmp_df_dict_values, keys=None, axis=1, sort="False")
        res_df_targets = pd.concat(tmp_df_dict_targets, keys=None, axis=1, sort="False")

        # get rid of multidimensional key inherited from concatenating a dictionary...
        res_df_values.columns = res_df_values.columns.droplevel(0)
        res_df_targets.columns = res_df_targets.columns.droplevel(0)

        # drop the last row of the values, to ensure equal size. This is crucial
        res_df_values = res_df_values[:-1]
        assert res_df_values.shape[0] == res_df_targets.shape[0], "DataLoader, Values and Targets unequal row sizes"

        # Drop NaN Rows and Columns, Replace remaining NaNs
        res_df_values[res_df_values.filter(regex="views").columns] = res_df_values.filter(regex='views').fillna(0)
        res_df_values = res_df_values.dropna(axis=1)
        res_df_values = res_df_values.dropna(axis=0)
        res_df_values = res_df_values.fillna(0)
        res_df_targets = res_df_targets.fillna(0)

        # The output dataframes have to have no NaNs. Assert that
        assert res_df_values.isna().sum().sum() == 0, "DataLoader, dataframe values has NaN"
        assert res_df_targets.isna().sum().sum() == 0, "DataLoader, dataframe targets has NaN"

        res_df_values.set_index(pd.DatetimeIndex(res_df_values.index), inplace=True)
        res_df_targets.set_index(pd.DatetimeIndex(res_df_targets.index), inplace=True)

        #print(res_df_values.dtypes.nunique())
        #exit(42)

        return res_df_values, res_df_targets

    def load_internal_data(self, p_stock_instead_of_efts=True):
        """
        Generates data frames that contain the available data from the internal stock source
        Returns two dataframes, each indexed by date:
        - First: Dataframe that contains the information, excluding the target columns
        - And: Dataframe that contains the target values, shifted by one day (the values that are to be predicted)
        :param p_stock_instead_of_efts: Set to true by default.
        :return: a dataframe that contains the values and a dataframe that contains the targets. Both indexed by date.
        """
        ctr = 0
        df_input_values = {}  # the dictionary to hold all data frames with their Values
        df_input_targets = {}  # the dictionary that holds the corresponding Targets

        # Depending on whether this is to fetch stock of efts data, adjust the directory
        if p_stock_instead_of_efts:
            tmp_complete_directory = self.data_directory_general_internal + self.data_subdirectory_stocks
        else:
            tmp_complete_directory = self.data_directory_general_internal + self.data_subdirectory_etf

        for filename in os.listdir(tmp_complete_directory):
            ctr += 1
            # Read the file into a dataframe and process it...
            if os.stat(tmp_complete_directory + filename).st_size > 0 \
                    and (ctr % self.every_xth_data == 0 or filename.__contains__("aapl") or
                         filename.__contains__("googl")) and filename.endswith(".txt"):
                df_input_values[filename] = pd.read_csv(tmp_complete_directory + filename, parse_dates=True,
                                                        index_col="Date")

                # OpenInt is very often just all zeros.
                del df_input_values[filename]['OpenInt']

                # Resample missing values, drop irrelevant rows
                df_input_values[filename] = df_input_values[filename].resample(
                    self.time_period.freq).asfreq().interpolate()
                df_input_values[filename] = df_input_values[filename][
                                            self.time_period[0].start_time:self.time_period[-1].end_time]

                # For unique column names, rename them (by appending filename)
                if self.rename_columns:
                    df_input_values[filename] = df_input_values[filename].rename(index=str,
                                                                                 columns={"Open": filename + "_Open",
                                                                                          "High": filename + "_High",
                                                                                          "Low": filename + "_Low",
                                                                                          "Close": filename + "_Close",
                                                                                          "Volume": filename + "_Volume",
                                                                                          "OpenInt": filename + "_OpenInt"})

                # Split the dataframe into the targets and into the values
                df_input_targets[filename] = pd.DataFrame()
                if not self.select_only_few_for_targets or \
                        (filename.__contains__("googl") or filename.__contains__("aapl")):  # A => B iff !A or B
                    for i in self.target:
                        if self.rename_columns:
                            tmp_rename_columns_str = filename + "_" + i
                        else:
                            tmp_rename_columns_str = i

                        if self.rename_target:
                            tmp_rename_target_str = filename + "_" + 'Target'
                        else:
                            tmp_rename_target_str = tmp_rename_columns_str

                        # Now perform the actual 'splitting' with the established column names...
                        df_input_targets[filename][tmp_rename_target_str] = df_input_values[
                            filename][tmp_rename_columns_str].shift(-1)
                        df_input_targets[filename].dropna(inplace=True)  # Remove last value
                        #del df_input_values[filename][tmp_rename_columns_str]

        # print("Done reading all csv files seperately... creating a large combined df now")
        res_df_values = pd.concat(df_input_values, keys=None, axis=1, sort="False")
        res_df_targets = pd.concat(df_input_targets, keys=None, axis=1, sort="False")

        # Possibly get rid of  NaN values, if desired
        if self.replace_nan_with_mean:
            res_df_values = res_df_values.fillna(res_df_values.mean())
            res_df_targets = res_df_targets.fillna(res_df_targets.mean())

        # get rid of multidimensional key inherited from concatenating a dictionary...
        res_df_values.columns = res_df_values.columns.droplevel(0)
        res_df_targets.columns = res_df_targets.columns.droplevel(0)

        return res_df_values, res_df_targets

    def load_external_wkpd_data(self):
        """
        Generates a dataframe that contains the available information in the preset time from external
        wkpd data sources.
        Note that this can not contain a target column.
        Returns one dataframe, indexed by date, containing available data
        :return: a dataframe that contains the values and a dataframe that contains the targets. Both indexed by date.
        """
        df_input_values = {}  # the dictionary to hold all data frames with their Values

        # The complete directory
        tmp_complete_directory = self.data_directory_general_external + self.data_subdirectory_wkpd

        for filename in os.listdir(tmp_complete_directory):
            # Read the file into a dataframe and process it...
            if os.stat(tmp_complete_directory + filename).st_size > 0 and filename.endswith(".csv"):
                df_input_values[filename] = pd.read_csv(tmp_complete_directory + filename, parse_dates=True,
                                                        index_col='timestamp', encoding="UTF-8")

                df_input_values[filename].index = pd.to_datetime(pd.Series(df_input_values[filename].index),
                                                                 format='%Y%m%d%H')

                # change frequency to Days, drop irrelevant rows
                df_input_values[filename] = df_input_values[filename].resample(self.time_period.freq).pad()
                df_input_values[filename] = df_input_values[filename][
                                            self.time_period[0].start_time:self.time_period[-1].end_time]

                # For unique column names, rename them (by appending filename)
                if self.rename_columns:
                    df_input_values[filename] = df_input_values[filename].rename(index=str,
                                                                                 columns={
                                                                                     "project": filename + "_project",
                                                                                     "article": filename + "_article",
                                                                                     "granularity": filename + "_granularity",
                                                                                     "access": filename + "_access",
                                                                                     "agent": filename + "_agent",
                                                                                     "views": filename + "_views"
                                                                                 })

        # print("Done reading all csv files seperately... creating a large combined df now")
        res_df_values = pd.concat(df_input_values, keys=None, axis=1, sort="False")

        # Possibly get rid of  NaN values, if desired
        if self.replace_nan_with_mean:
            res_df_values = res_df_values.fillna(res_df_values.mean())

        # get rid of multidimensional key inherited from concatenating a dictionary...
        res_df_values.columns = res_df_values.columns.droplevel(0)

        # FIXME:
        res_df_values = res_df_values.fillna(0)

        return res_df_values

    def load_external_ggl_trnds_data(self):
        """
        Generates a dataframe that contains the available information in the preset time from external
        ggl trnd data sources.
        Note that this can not contain a target column.
        Returns one dataframe, indexed by date, containing available data
        :return: a dataframe that contains the values and a dataframe that contains the targets. Both indexed by date.
        """
        df_input_values = {}  # the dictionary to hold all data frames with their Values

        # The complete directory
        tmp_complete_directory = self.data_directory_general_external + self.data_subdirectory_ggl_trnds

        for filename in os.listdir(tmp_complete_directory):
            # Read the file into a dataframe and process it...
            if os.stat(tmp_complete_directory + filename).st_size > 0 and filename.endswith(".csv"):
                df_input_values[filename] = pd.read_csv(tmp_complete_directory + filename, parse_dates=True,
                                                        index_col="Month", skiprows=2)

                # Add missing values to all days, Drop irrelevant rows  (csv frequency is Months, not Days)
                df_input_values[filename] = df_input_values[filename].resample(self.time_period.freq).pad()
                df_input_values[filename] = df_input_values[filename][
                                            self.time_period[0].start_time:self.time_period[-1].end_time]

                # For unique column names, rename them (by appending filename)
                if self.rename_columns:
                    df_input_values[filename] = df_input_values[filename].rename(index=str,
                                                                                 columns={
                                                                                     "aapl": filename + "_aapl"
                                                                                 })

        # print("Done reading all csv files seperately... creating a large combined df now")
        res_df_values = pd.concat(df_input_values, keys=None, axis=1, sort="False")

        # Replace "<1"
        res_df_values = res_df_values.replace({"<1": "0.2"}, regex=True)

        # Possibly get rid of  NaN values, if desired
        if self.replace_nan_with_mean:
            res_df_values = res_df_values.fillna(res_df_values.mean())

        # get rid of multidimensional key inherited from concatenating a dictionary...
        res_df_values.columns = res_df_values.columns.droplevel(0)
        res_df_values['fb: (Worldwide)'] = pd.to_numeric(res_df_values['fb: (Worldwide)'])

        return res_df_values

    """
    deprecated? (legacy function)
    """

    def load_single_stock(self, file: str):
        """
        This method loads the data
        :param time_period: use only the data from the given time period
        :return: a dataframe containing the data
        """
        self.internal_data = pd.read_csv(self.data_directory_general_internal + self.data_subdirectory_stocks + file,
                                         index_col='Date', parse_dates=True)
        self.internal_data['Target'] = self.internal_data['Close'].shift(-1)  # Add target
        self.internal_data.dropna(inplace=True)  # Remove last value
        return self.internal_data
