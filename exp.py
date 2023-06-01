# Installations
# aicrowd-cli for downloading challenge data and making submissions
# pyarrow for saving to parquet for submissions

# !pip install aicrowd-cli pyarrow


# Login to AIcrowd and download the data 

# !aicrowd login
# !aicrowd dataset download --challenge task-1-next-product-recommendation


# Setup data and task information

import os
import numpy as np
import pandas as pd
from functools import lru_cache

train_data_dir = '.'
test_data_dir = '.'
task = 'task1'
PREDS_PER_SESSION = 100

# Cache loading of data for multiple calls

@lru_cache(maxsize=1)
def read_product_data():
    return pd.read_csv(os.path.join(train_data_dir, 'products_train.csv'))

@lru_cache(maxsize=1)
def read_train_data():
    return pd.read_csv(os.path.join(train_data_dir, 'sessions_train.csv'))

@lru_cache(maxsize=3)
def read_test_data(task):
    return pd.read_csv(os.path.join(test_data_dir, f'sessions_test_{task}.csv'))


# Data Description

# The Multilingual Shopping Session Dataset is a collection of anonymized customer sessions containing products from six different locales, namely English, German, Japanese, French, Italian, and Spanish. 
# It consists of two main components: user sessions and product attributes. User sessions are a list of products that a user has engaged with in chronological order, while product attributes include various details like product title, price in local currency, brand, color, and description.
# Each product as its associated information:

# locale: the locale code of the product (e.g., DE)

# id: a unique for the product. Also known as Amazon Standard Item Number (ASIN) (e.g., B07WSY3MG8)

# title: title of the item (e.g., “Japanese Aesthetic Sakura Flowers Vaporwave Soft Grunge Gift T-Shirt”)

# price: price of the item in local currency (e.g., 24.99)

# brand: item brand name (e.g., “Japanese Aesthetic Flowers & Vaporwave Clothing”)

# color: color of the item (e.g., “Black”)

# size: size of the item (e.g., “xxl”)

# model: model of the item (e.g., “iphone 13”)

# material: material of the item (e.g., “cotton”)

# author: author of the item (e.g., “J. K. Rowling”)

# desc: description about a item’s key features and benefits called out via bullet points (e.g., “Solid colors: 100% Cotton; Heather Grey: 90% Cotton, 10% Polyester; All Other Heathers …”)

def read_locale_data(locale, task):
    products = read_product_data().query(f'locale == "{locale}"')
    sess_train = read_train_data().query(f'locale == "{locale}"')
    sess_test = read_test_data(task).query(f'locale == "{locale}"')
    return products, sess_train, sess_test


def show_locale_info(locale, task):
    products, sess_train, sess_test = read_locale_data(locale, task)
    
    train_l = sess_train['prev_items'].apply(lambda sess: len(sess))
    test_l = sess_test['prev_items'].apply(lambda sess: len(sess))
    
    print(f"Locale: {locale} \n"
            f"Number of products: {products['id'].nunique()} \n"
            f"Number of train sessions: {len(sess_train)} \n"
            f"Train session lengths - "
            f"Mean: {train_l.mean():.2f} | Median {train_l.median():.2f} | "
            f"Min: {train_l.min():.2f} | Max {train_l.max():.2f} \n"
            f"Number of test sessions: {len(sess_test)}"
        )
    if len(sess_test) > 0:
        print(
                f"Test session lengths - "
            f"Mean: {test_l.mean():.2f} | Median {test_l.median():.2f} | "
            f"Min: {test_l.min():.2f} | Max {test_l.max():.2f} \n"
        )
    print("======================================================================== \n")

products = read_product_data()
locale_names = products['locale'].unique()
for locale in locale_names:
    show_locale_info(locale, task)



# Generate Submission
# Submission format:

# The submission should be a parquet file with the sessions from all the locales.
# Predicted products ids per locale should only be a valid product id of that locale.
# Predictions should be added in new column named "next_item_prediction".
# Predictions should be a list of string id values

def random_predicitons(locale, sess_test_locale):
    random_state = np.random.RandomState(42)
    products = read_product_data().query(f'locale == "{locale}"')
    predictions = []
    for _ in range(len(sess_test_locale)):
        predictions.append(
            list(products['id'].sample(PREDS_PER_SESSION, replace=True, random_state=random_state))
        ) 
    sess_test_locale['next_item_prediction'] = predictions
    sess_test_locale.drop('prev_items', inplace=True, axis=1)
    return sess_test_locale

test_sessions = read_test_data(task)
predictions = []
test_locale_names = test_sessions['locale'].unique()

for locale in test_locale_names:
    sess_test_locale = test_sessions.query(f'locale == "{locale}"').copy()
    predictions.append(
        random_predicitons(locale, sess_test_locale)
    )

predictions = pd.concat(predictions).reset_index(drop=True)
predictions.sample(5)


# Validate predictions

def check_predictions(predictions, check_products=False):
    """
    These tests need to pass as they will also be applied on the evaluator
    """
    test_locale_names = test_sessions['locale'].unique()
    for locale in test_locale_names:
        sess_test = test_sessions.query(f'locale == "{locale}"')
        preds_locale =  predictions[predictions['locale'] == sess_test['locale'].iloc[0]]
        assert sorted(preds_locale.index.values) == sorted(sess_test.index.values), f"Session ids of {locale} doesn't match"

        if check_products:
            # This check is not done on the evaluator
            # but you can run it to verify there is no mixing of products between locales
            # Since the ground truth next item will always belong to the same locale
            # Warning - This can be slow to run
            products = read_product_data().query(f'locale == "{locale}"')
            predicted_products = np.unique( np.array(list(preds_locale["next_item_prediction"].values)) )
            assert np.all( np.isin(predicted_products, products['id']) ), f"Invalid products in {locale} predictions"

# Its important that the parquet file you submit is saved with pyarrow backend
predictions.to_parquet(f'submission_{task}.parquet', engine='pyarrow')


# Submit to AIcrowd

# You can submit with aicrowd-cli, or upload manually on the challenge page.
# !aicrowd submission create -c task-1-next-product-recommendation -f "submission_task1.parquet"
