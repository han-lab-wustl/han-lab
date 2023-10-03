
import pandas as pd
import os 

def fixcsvcols(csv):
    if type(csv) == str:
        df = pd.read_csv(csv)
        cols=[[xx+"_x",xx+"_y",xx+"_likelihood"] for xx in pd.unique(df.iloc[0]) if xx!="bodyparts"]
        cols = [yy for xx in cols for yy in xx]; cols.insert(0, 'bodyparts')
        df.columns = cols
        df=df.drop([0,1])
        savecsv = csv[:-4]+'_fixcol.csv'
        df.to_csv(savecsv)
    else:
        print("\n ******** please pass path to csv ********")
    return df

if __name__ == "__main__":
    # path to csv from dlc
    csv = r'Y:\DLC\dlc_mixedmodel2\230505_E190DLC_resnet50_MixedModel_trial_2Mar27shuffle1_750000.csv'
    # run function
    df = fixcsvcols(csv)