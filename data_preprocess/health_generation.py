import zipfile
import numpy as np
import pandas as pd
from os import path
import matplotlib.pyplot as plt
import seaborn as sns

# perpare the orginal dataset "HHP_release3.zip" to our used data

def main():
    data_file = path.join('dataset', 'HHP_release3.zip')
    zf = zipfile.ZipFile(data_file)

    df_claims = pd.read_csv(zf.open('Claims.csv'), sep=',')
    df_drugs = pd.read_csv(zf.open('DrugCount.csv'), sep=',')
    df_labs = pd.read_csv(zf.open('LabCount.csv'), sep=',')
    df_members = pd.read_csv(zf.open('Members.csv'), sep=',')

    df_claims_new = preprocess_claims(df_claims)
    df_drugs_new = preprocess_drugs(df_drugs)
    df_labs_new = preprocess_labs(df_labs)
    df_members_new = preprocess_members(df_members)

    df_labs_drugs = pd.merge(df_labs_new, df_drugs_new, on=['MemberID'], how='outer')
    df_labs_drugs_claims = pd.merge(df_labs_drugs, df_claims_new, on=['MemberID'], how='outer')
    df_health = pd.merge(df_labs_drugs_claims, df_members_new, on=['MemberID'], how='outer')

    count_nulls(df_health)
    
    df_health.fillna(0, inplace=True)
    print('df_health.shape:', df_health.shape)

    health_file = './dataset/health_without_year.csv'
    df_health.to_csv(health_file, index=False)
    return df_health





def preprocess_claims(df_claims):

    df_claims = df_claims[['MemberID','Year','Specialty','PayDelay','PrimaryConditionGroup','CharlsonIndex','ProcedureGroup']]
    df_claims.loc[df_claims['PayDelay'] == '162+', 'PayDelay'] = 162
    df_claims['PayDelay'] = df_claims['PayDelay'].astype(int)

    df_claims.loc[df_claims['CharlsonIndex'] == '0', 'CharlsonIndex'] = 0
    df_claims.loc[df_claims['CharlsonIndex'] == '1-2', 'CharlsonIndex'] = 2
    df_claims.loc[df_claims['CharlsonIndex'] == '3-4', 'CharlsonIndex'] = 4
    df_claims.loc[df_claims['CharlsonIndex'] == '5+', 'CharlsonIndex'] = 6
    df_claims['CharlsonIndex'] = df_claims['CharlsonIndex'].astype(int)

    claims_cat_names = ['Specialty','PrimaryConditionGroup','ProcedureGroup']
    for cat_name in claims_cat_names:
        df_claims[cat_name].fillna(f'{cat_name}_?', inplace=True)
    df_claims = pd.get_dummies(df_claims, columns=claims_cat_names, prefix_sep='=')

    oh = [col for col in df_claims if '=' in col]

    agg = { 'CharlsonIndex': 'max',
        'PayDelay': ['sum', 'max', 'min']
    }
    for col in oh:
        agg[col] = 'sum'

    df_claims.drop(columns=['Year'], inplace=True)

    df_group = df_claims.groupby(['MemberID'])
    df_claims = df_group.agg(agg).reset_index()
    df_claims.columns = [ 'MemberID', 'max_CharlsonIndex',
                    'PayDelay_total', 'PayDelay_max', 'PayDelay_min'
                        ] + oh
    print('df_claims.shape = ', df_claims.shape)
    return df_claims


def preprocess_drugs(df_drugs):
    df_drugs.drop(columns=['Year'], inplace=True)
    df_drugs.drop(columns=['DSFS'], inplace=True)
    df_drugs['DrugCount'] = df_drugs['DrugCount'].apply(lambda x: int(x.replace('+', '')))
    df_drugs = df_drugs.groupby(['MemberID']).agg({'DrugCount': ['sum', 'count']}).reset_index()
    df_drugs.columns = [ 'MemberID', 'DrugCount_total', 'DrugCount_months']
    print('df_drugs.shape = ', df_drugs.shape)
    return df_drugs

def preprocess_labs(df_labs):
    df_labs.drop(columns=['Year'], inplace=True)
    df_labs.drop(columns=['DSFS'], inplace=True)
    df_labs['LabCount'] = df_labs['LabCount'].apply(lambda x: int(x.replace('+', '')))
    df_labs = df_labs.groupby(['MemberID']).agg({'LabCount': ['sum', 'count']}).reset_index()
    df_labs.columns = [ 'MemberID', 'LabCount_total', 'LabCount_months']
    print('df_labs.shape = ', df_labs.shape)
    return df_labs

def preprocess_members(df_members):
    df_members['AgeAtFirstClaim'].fillna('?', inplace=True)
    df_members['Sex'].fillna('?', inplace=True)
    print('df_members.shape = ', df_members.shape)
    return df_members


def count_nulls(df):
    null_counter = df.isnull().sum(axis=0)
    null_counter = null_counter[null_counter > 0]
    null_percent = df.isnull().sum(axis=0) / df.shape[0] * 100
    null_percent = null_percent[null_percent > 0]
    null_df = pd.concat([null_counter,null_percent],axis=1)
    null_df.columns = ['count','percent']
    print(null_df)