import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def basic_age_analysis(df):
    mean_age = df['age'].mean()
    std_dev_age = df['age'].std()
    age_array = df['age'].to_numpy()
    min_age = min(age_array)
    max_age = max(age_array)
    result = pd.DataFrame({
        'mean_age': [mean_age],
        'std_dev_age': [std_dev_age],
        'min_age': [min_age],
        'max_age': [max_age]
    })
    return result

def basic_mr_sessions_analysis(df): #average number of mr sessions with min and max
    mean_mris = df['mr_sessions'].mean()
    std_dev_mris = df['mr_sessions'].std()
    mris_array = df['mr_sessions'].to_numpy()
    min_mris = min(mris_array)
    max_mris = max(mris_array)
    result = pd.DataFrame({
        'mean_mris': [mean_mris],
        'std_dev_mris': [std_dev_mris],
        'min_mris': [min_mris],
        'max_mris': [max_mris]
    })
    return result

def basic_duration_analysis(df):
    mean_dur = df['duration'].mean()
    std_dev_dur = df['duration'].std()
    dur_array = df['duration'].to_numpy()
    min_dur = min(dur_array)
    max_dur = max(dur_array)
    results = pd.DataFrame({
        'mean_duration': [mean_dur],
        'std_dev_duration': [std_dev_dur],
        'min_duration': [min_dur],
        'max_duration': [max_dur]
    })
    return results

def basic_race_analysis(df):
    asian = df[df['race']=='Asian'].shape[0]
    african_american = df[df['race']=='African American'].shape[0]
    caucasian = df[df['race']=='Caucasian'].shape[0]
    no_info = df[pd.isna(df['race'])].shape[0]
    results = pd.DataFrame({
        'asian': [asian],
        'african_american': [african_american],
        'caucasian': [caucasian],
        'no_info': [no_info]
    })
    return results

def plot_age_histograms(df1, df2, df3, num_bins=30):
    plt.figure(figsize=(10, 6))

    sns.histplot(df1['age'].dropna(), bins=num_bins, color='blue', label='All participants', alpha=0.5)
    sns.histplot(df2['age'].dropna(), bins=num_bins, color='red', label='Male', alpha=0.5)
    sns.histplot(df3['age'].dropna(), bins=num_bins, color='green', label='Female', alpha=0.5)

    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.title('Age Distribution By Sex')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
    plt.savefig('results/age_histogram.png')

def plot_mri_info_by_age(df, age_th1 = 50, age_th2 = 65, age_th3 = 75, age_th4 = 90):
    age_group1 = df[df['age'] <= age_th1]
    age_group2 = df[(age_th1 < df['age']) & (df['age'] <= age_th2)]
    age_group3 = df[(age_th2 < df['age']) & (df['age'] <= age_th3)]
    age_group4 = df[(age_th3 < df['age']) & (df['age'] <= age_th4)]
    age_group5 = df[age_th4<df['age']]
    mean_dur1 = age_group1['duration'].mean()
    mean_dur2 = age_group2['duration'].mean()
    mean_dur3 = age_group3['duration'].mean()
    mean_dur4 = age_group4['duration'].mean()
    mean_dur5 = age_group5['duration'].mean()
    mean_sessions1 = age_group1['mr_sessions'].mean()
    mean_sessions2 = age_group2['mr_sessions'].mean()
    mean_sessions3 = age_group3['mr_sessions'].mean()
    mean_sessions4 = age_group4['mr_sessions'].mean()
    mean_sessions5 = age_group5['mr_sessions'].mean()
    means_duration = [mean_dur1, mean_dur2, mean_dur3, mean_dur4, mean_dur5]
    means_sessions = [mean_sessions1, mean_sessions2, mean_sessions3, mean_sessions4, mean_sessions5]
    labels = [
        "<=" + str(age_th1),
        str(age_th1+1) + "-" + str(age_th2),
        str(age_th2+1) + "-" + str(age_th3),
        str(age_th3+1) + "-" + str(age_th4),
        ">" + str(age_th4+1)
    ]
    #plot1
    plt.figure(figsize=(8, 6))
    plt.bar(labels, means_duration, color='skyblue')
    plt.xlabel("Age")
    plt.ylabel("Mean duration")
    plt.title("Mean duration between baseline and final follow-up MRI scan by Age Group")
    plt.xticks(rotation=45)
    plt.show()
    plt.savefig('results/duration_by_age.png')
    #plot2
    plt.figure(figsize=(8, 6))
    plt.bar(labels, means_sessions, color='skyblue')
    plt.xlabel("Age")
    plt.ylabel("Mean number of MRI scans")
    plt.title("Mean MRI Scans by Age Group")
    plt.xticks(rotation=45)
    plt.show()
    plt.savefig('results/mris_by_age.png')
    age_group1 = df[df['age'] <= age_th1]
    age_group2 = df[(age_th1 < df['age']) & (df['age'] <= age_th2)]
    age_group3 = df[(age_th2 < df['age']) & (df['age'] <= age_th3)]
    age_group4 = df[(age_th3 < df['age']) & (df['age'] <= age_th4)]
    age_group5 = df[age_th4<df['age']]
    mean_dur1 = age_group1['mr_sessions'].mean()
    mean_dur2 = age_group2['mr_sessions'].mean()
    mean_dur3 = age_group3['mr_sessions'].mean()
    mean_dur4 = age_group4['mr_sessions'].mean()
    mean_dur5 = age_group5['mr_sessions'].mean()
    means = [mean_dur1, mean_dur2, mean_dur3, mean_dur4, mean_dur5]
    labels = [
        "<=" + str(age_th1),
        str(age_th1+1) + "-" + str(age_th2),
        str(age_th2+1) + "-" + str(age_th3),
        str(age_th3+1) + "-" + str(age_th4),
        ">" + str(age_th4+1)
    ]
    plt.figure(figsize=(8, 6))
    plt.bar(labels, means, color='skyblue')
    plt.xlabel("Age")
    plt.ylabel("Mean number of MRI scans")
    plt.title("Mean MRI Scans by Age Group")
    plt.xticks(rotation=45)
    plt.show()
    plt.savefig('results/mris_by_age.png')

def plot_race(df):
    asian = df[df['race']=='Asian'].shape[0]
    african_american = df[df['race']=='African American'].shape[0]
    caucasian = df[df['race']=='Caucasian'].shape[0]
    no_info = df[pd.isna(df['race'])].shape[0]
    numbers = [asian,african_american,caucasian,no_info]
    labels = ["Asian","African-American","Caucasian","No Info"]
    plt.figure(figsize=(8, 6))
    plt.bar(labels, numbers, color='skyblue')
    plt.xlabel("Race")
    plt.ylabel("Number of Subjects")
    plt.title("Number of Subjects by Race")
    plt.xticks(rotation=45)
    plt.show()
    plt.savefig('results/race_plot.png')

def extract_race_counts(df):
    asian = df[df['race']=='Asian'].shape[0]
    african_american = df[df['race']=='African American'].shape[0]
    caucasian = df[df['race']=='Caucasian'].shape[0]
    no_info = df[pd.isna(df['race'])].shape[0]
    numbers = [asian,african_american,caucasian,no_info]
    labels = ["Asian","African-American","Caucasian","No Info"]
    return labels, numbers

def basic_education_analysis(df):
    unique_edu = df['education'].unique()
    unique_edu = np.sort(unique_edu)
    counts = np.zeros(len(unique_edu))
    j = 0
    for edu_level in unique_edu:
        if pd.isna(edu_level):
            counts[j] = df[pd.isna(df['education'])].shape[0]
        else:
            counts[j] = df[df['education']==edu_level].shape[0]
        j += 1
    unique_edu = np.where(pd.isna(unique_edu), "No info", unique_edu.astype(str))
    return unique_edu, counts

def plot_education(df):
    labels, numbers = basic_education_analysis(df)
    plt.figure(figsize=(8, 8))
    plt.bar(labels, numbers, color='skyblue')
    plt.xlabel("Education")
    plt.ylabel("Number of Subjects")
    plt.title("Number of Subjects by Education")
    plt.xticks(rotation=45)
    plt.show()
    plt.savefig('results/education_plot.png')

def extract_counts_for_mris(df):
    unique_num_of_scans = df['mr_sessions'].unique()
    unique_num_of_scans = np.sort(unique_num_of_scans)
    counts = np.zeros(len(unique_num_of_scans))
    i = 0
    for num_scans in unique_num_of_scans:
        counts[i] = df[df['mr_sessions']==num_scans].shape[0]
        i += 1
    return unique_num_of_scans, counts