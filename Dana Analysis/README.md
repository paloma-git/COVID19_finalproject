File: analysis_symptomatic.py
This python file loads two datasets, from two different locations (Israel=IL & Mexico city=MX).
the datasets were initially cleaned to a level where they are comparable. It contains 7 features, 5 of which are COVID-19 symptoms (Cough, Fever, Sore throat, Shortness of breath, Head ache) and gender (if women=1) and age group (if over 60=1).
All samples are positive to COVID-19.
The code then removes all asymptomatic samples.
Then, statistical tests are performed and plotted (Pearson correlation coefficient, Jaccard index).
Finaly, a k-means clustering is performed.


