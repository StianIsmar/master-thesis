'''
This class contains all of the 10 second intervals related to a specific wind turbine. In this case, this is one of the
four wind turbines at Skomakerfjellet.
'''
class Wt_data():
    def __init__(self):
        self.dataframes = []

    def add_dataframe(self, dataframe):
        self.dataframes.append(dataframe)

wt_01 = Wt_data()


wt_02 = Wt_data()


wt_03 = Wt_data()


wt_04 = Wt_data()



