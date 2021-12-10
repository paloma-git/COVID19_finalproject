# COVID19_finalproject
This is the repository that contains the code used for our statistical and clustering analyses for CS 791

Developed by: Paloma Cepeda and Nadav Zimron-Politi

The following datasets were used for analysis:

Mexico City Dataset: https://datos.cdmx.gob.mx/dataset/base-covid-sinave
(downloaded: October 12, 2021; last accessed: December 10, 2021)

Israel Dataset: https://data.gov.il/dataset/covid-19
(downloaded: October 27, 2021; last accessed: December 10, 2021)

The "Data Cleanup" folder includes the codes used to reduce the number of samples and features presented in the raw data obtained above.
The datasets were  cleaned to a level where they are comparable. They each contain 7 features, 5 of which are COVID-19 symptoms (Cough, Fever, Sore throat, Shortness of breath, Headache) and gender (if gender is female = 1) and age group (if age over 60 = 1). All samples are of patients who tested positive for COVID-19.

The "Data Analysis" folder uses the already cleaned-up datasets ready to use with the provided codes.
The clean datasets are: 
  - Israel_dataset_clean.csv
  - clean_data_mx.csv

The analysis codes are:
  - analysis_symptomatic.py: this program removes asymptomatic patients from each dataset and performs statistical tests (Pearson correlation coefficient, Jaccard index), as well as k-means clustering
  - analysis_symptomaticANDsymptomatic.py: this program performs the same statistical tests as above, but includes asymptomatic patients in the analysis
  - kmeans_PCA.py: this program performs k-means clustering analysis on patients suffering from COVID-19 who are symptomatic. It also analyses the resulting clusters and performs Principal Component Analysis (PCA) to reduce the dimensionality for easy visualization of these clusters.


Contact information:

Paloma Cepeda: pcepeda@nevada.unr.edu 

Nadav Zimron-Politi: nzimronpoliti@nevada.unr.edu

Please contact us for any issues or suggestions. Thank you.
