# Classifier_Meal_NoMeal
Trains a machine model to assess whether a person has eaten a meal or not eaten a meal.  A training data set is provided. 

Directions

Meal data is extracted as follows:

From the InsulinData.csv file, the column Y for a non NAN non zero value. This time indicates the start of meal consumption time tm. Meal data comprises a 2hr 30 min stretch of CGM data that starts from tm-30min and extends to tm+2hrs. 

No meal data comprises 2 hrs of raw data that does not have meal intake. 

Extraction: Meal data 

Start of a meal can be obtained from InsulinData.csv. Search column Y for a non NAN non zero value. This time indicates the start of a meal. There can be three conditions:

There is no meal from time tm to time tm+2hrs. Then use this stretch as meal data
There is a meal at some time tp in between tp>tm and tp< tm+2hrs. Ignore the meal data at time tm and consider the meal at time tp instead. 
There is a meal at time tm+2hrs, then consider the stretch from tm+1hr 30min to tm+4hrs as meal data.
Extraction: No Meal data

Start of no meal is at time tm+2hrs where tm is the start of some meal. We need to obtain a 2 hr stretch of no meal time. So you need to find all 2 hr stretches in a day that have no meal and do not fall within 2 hrs of the start of a meal. 

Feature Extraction and Selection:

Fourier Transformation, RMS and CGM, time differential are considered as features. These features amplified the difference between Meal and No Meal data points.

Test Data:

The test data is a matrix of size N×24, where N is the total number of tests and 24 is the size of the CGM time series. N will have some distribution of meal and no meal data. 

Note here that for meal data you are asked to obtain a 2 hr 30 min time series data, while for no meal you are taking 2 hr. However, a machine will not take data with different lengths. Hence, in the feature extraction step, you have to ensure that features extracted from both meal and no meal data have the same length. 

Output format:

A N×1 vector of 1s and 0s, where if a row is determined to be meal data, then the corresponding entry will be 1, and if determined to be no meal, 
the corresponding entry will be 0.

This vector will be saved in a Result.csv file. 

Given: 	

Meal Data and No Meal Data of subject 1 and 2

Ground truth labels of Meal and No Meal for subject 1 and 2	 

Using Python, a machine model is trained to recognize whether a sample in the training data set represents a person who has eaten (Meal), or not eaten (No Meal). 

The following tasts has been performed:

Extract features from Meal and No Meal training data set. 
Feature Extraction and Selection
Train a machine to recognize Meal or No Meal data.
Used k fold cross validation on the training data to evaluate your recognition system.
 
