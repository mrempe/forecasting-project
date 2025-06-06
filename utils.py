# utils.py

from statsmodels.tsa.stattools import adfuller # Augmented Dickey-Fuller Test for stationarity check
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

def perform_adf_test(data):
    # Perform ADF test
    result = adfuller(data)

    # Print the results
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])

    # Check if the data is stationary based on the p-value
    if result[1] <= 0.05:
        print("The data is stationary")
    else:
        print("The data is not stationary, Data can be processed further")

# Example Usage
#perform_adf_test(df['Close'])


def plot_differencing_acf_pacf(data):
    fig, axes = plt.subplots(3, 2, sharex=True)

    # Original Series
    axes[0, 0].plot(data)
    axes[0, 0].set_title('Original Series')
    plot_acf(data, ax=axes[0, 1])

    # 1st Differencing
    diff_1 = data.diff().dropna()
    axes[1, 0].plot(diff_1)
    axes[1, 0].set_title('1st Order Differencing')
    plot_acf(diff_1, ax=axes[1, 1])

    # 2nd Differencing
    diff_2 = data.diff().diff().dropna()
    axes[2, 0].plot(diff_2)
    axes[2, 0].set_title('2nd Order Differencing')
    plot_acf(diff_2, ax=axes[2, 1])

    plt.tight_layout()
    plt.show(block=False)

# Example Usage
# plot_differencing_acf_pacf(df['Close'])