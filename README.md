# Regression model
Machine learning

You are to produce a family of regression models that predict the response of massive online social network (MOSN) users, further referred to as subjects, to anonymous grades given to their profile pictures.

Odnoklassniki.ru ("classmates") was the largest Russian-language MOSN at the time of data collection, a "Russian Facebook." 3000 subjects with known self-declared age and gender were randomly selected for the experiment. Each subject's profile picture was randomly rated by the researcher's avatar on the scale of 1 ("dislike") to 6 ("like very much"). We call this rating a stimulus grade. In response, some subjects rated the avatar's profile picture on the same scale. We call this rating the response grade. The data collected in the experiment are on GitHub. The first and the second columns have the subject's gender and age, the third column has the response grade, and the fourth column has the stimulus grade. Note that some of the response grades are missing.

Program:

Read the data file as a pandas data frame, eliminating the missing values.
Produce the following descriptive charts:.
Two histograms of the subject's ages, one for female subjects and one for male subjects, both in the same chart (superimposed with .75 transparency)
A scatter plot showing the generosity (the difference between the response grade and the stimulus grade) vs age. The points shall be colored differently for male and female subjects. Since there are only few discrete levels of generosity (-5 through +5), most of the points in the chart will be obscured by other points. To make the chart more informative, the points shall be semitransparent and slightly randomly scattered around their actual positions. This can be accomplished by adding a small normally distributed random displacement to both X and Y coordinates of each point. The amount of displacement shall make it visible as many points as possible, but shall not blur the boundaries between the discrete generosity levels.
Split the data into a 70% training set and 30% validation (testing) set. The split must be random but reproducible.
Produce two data models: logistic regression with C=1 and Random Forest Classifier -- and fit each model using only the training set. Replace the gender with indicators (two dummy variables). Use age, gender, and stimulus grade as model features, and response grade as model outcome.
Calculate the score and confusion matrix of both models after applying the functions to the validation data set.
Your submission shall consist of the program code and a report. The report shall consist of at least two pages and include both charts (do not send the charts separately). The report shall describe the data set (size, number of NAs, distribution of subjects by gender, etc.); what transformations, if any, were applied to the variables; and what you learned from the charts. The report compare both models in terms of accuracy and contain a recommendation about which model shall be used to make accurate predictions.
