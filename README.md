# README #

This repository is mainly for music genre prediction.


To create environment, type:

pip install --user pipenv

pipenv install


## The main code for music genre predictions:
1. Step1_EDA_numeric_model.ipynb
2. Step2_Text_EDA_model.ipynb
3. Step3_combine.ipynb

The first file processes numeric features, model indicates a 63~64% accuracy with a cohen's kappa of around ~.57 if using numeric feature alone.

The second file processes text (tag and title) features,  model indicates a 40~44% accuracy with a cohen's kappa of around ~.31 if using numeric feature alone.

The third file contains both numeric and text features,  model indicates a 66~69% accuracy with a cohen's kappa of ~.60.

Note: due to time limits and computation power, grid search only tried a very limited set of parameters
## To run the file (test cases):
pipenv shell

export PYTHONPATH="/Users/daisy/Google Drive/DBS_Mini_Project" (your path)

pytest test_code/

###### due to time lmits, only wrote two system level test cases - from raw data to predictions.

## To run the FastAPI
Instructions are uploaded in film: "FastAPI - Screen Recording 2021-09-19 at 2.42.00 PM.mov"
Just in case if .mov does not work, I converted it to .mp4
 "FastAPI - Screen Recording 2021-09-19 at 2.42.00 PM.mp4"

The API will output the first 10 predictions for demonstration purposes

pipenv shell

option 1: run app.py

option 2: Type in terminal 

uvicorn app:app --reload


### tutorial about FastAPI
https://github.com/krishnaik06/FastAPI

### Contribution guidelines ###

* Writing tests
* Code review
