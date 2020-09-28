from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

# Gather Data
boston_dataset = load_boston()
data = pd.DataFrame(data=boston_dataset.data,
                    columns=boston_dataset.feature_names)
features = data.drop(['INDUS', 'AGE'], axis=1)

log_prices = np.log(boston_dataset.target)
target = pd.DataFrame(log_prices, columns=['PRICE'])

CRIME_IDX = 0
ZN_IDX = 1
CHAS_IDX = 2
RM_IDX = 4
PTRATIO_IDX = 8

property_stats = features.mean().values.reshape(1, 11)

regr = LinearRegression().fit(features, target)
fitted_vals = regr.predict(features)

MSE = mean_squared_error(target, fitted_vals)
RMSE = np.sqrt(MSE)


def get_log_estimate(nr_rooms,
                     students_per_classroom,
                     next_to_river,
                     high_confidence):
    # Configure property
    property_stats[0, RM_IDX] = nr_rooms
    property_stats[0, PTRATIO_IDX] = students_per_classroom

    if next_to_river:
        property_stats[0, CHAS_IDX] = 1
    else:
        property_stats[0, CHAS_IDX] = 0

    # Make prediction
    log_estimate = regr.predict(property_stats)

    # Calc range
    if high_confidence:
        upper_bound = log_estimate + 2 * RMSE
        lower_bound = log_estimate - 2 * RMSE
        interval = 95
    else:
        upper_bound = log_estimate + RMSE
        lower_bound = log_estimate - RMSE
        interval = 68

    return log_estimate[0, 0], upper_bound[0, 0], lower_bound[0, 0], interval


def get_dollar_estimate(rm,
                        ptratio,
                        chas=False,
                        large_range=True):
    """
    Estimate the price of a property in Boston.
    :param rm: number of rooms in the property
    :param ptratio: number of students per teacher in the classroom for the school in the area
    :param chas: True if the property is next to the river, False otherwise
    :param large_range: True for a 95% prediction interval, False for a 68% interval
    :return: prints out the price of the property in dollars
    """

    if (rm < 1) or (ptratio < 1):
        print('That is unrealistic')
        return

    zillow_median_price = 583.3
    scale_factor = zillow_median_price / np.median(boston_dataset.target)

    log_est, upper, lower, conf = get_log_estimate(rm,
                                                   ptratio,
                                                   chas,
                                                   large_range)

    dollar_est = np.exp(log_est) * 1000 * scale_factor
    dollar_hi = np.exp(upper) * 1000 * scale_factor
    dollar_low = np.exp(lower) * 1000 * scale_factor

    rounded_est = np.around(dollar_est, -3)
    rounded_hi = np.around(dollar_hi, -3)
    rounded_low = np.around(dollar_low, -3)

    print(f'The estimated property value is {rounded_est}.')
    print(f'At {conf}% confidence the valuation range is')
    print(f'USD {rounded_low} at the lower end to USD {rounded_hi} at the high end.')


print(get_dollar_estimate(0, 10))
