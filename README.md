# TrueTag
URL based News Article tag recommendation system using TF-IDF.

## Requirements

-    Python 3.7
-    Newspaper3k  0.2.8 (tested)
-    TQDM
-    Pandas       1.0.2 (tested)
-    Scikit-Learn
-    [All the news](https://www.kaggle.com/snapcrack/all-the-news) dataset used,
     its modified version can be downloaded from [here](https://drive.google.com/open?id=1QMK1A9ClSJ7VWqxxFrnfICM7K5SnoDZ2)

## Data Preprocessing and Model Training

To Preprocess data, 'main.py' takes 'train' as an argument
'train' consists of three conditional arguments:

-    Use flag '-nc' to skip data cleaning   
-    Use '-d [dataset_name]' to use custom dataset (to be put in TrueTag\dataset folder)(Default name: 'articles.csv')
-    Use '-t [number_of_rows_to_truncate_after] ' to truncate the dataset after a number of rows 

Example:

```sh
python main.py train -d articles.csv -t 100
```
## Prediction

To predict the output, 'main.py' takes 'predict' as an argument
'predict' consists of two positional arguments:

-    'n' is the number of tags to generate based on descending order of relevancy
-    'url' is the url of the news article.

Example:

```sh
python main.py predict 10 https://github.blog/2020-02-12-supercharge-your-command-line-experience-github-cli-is-now-in-beta/
```
