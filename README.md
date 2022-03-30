# Stock-Index-Prediction-Based-on-Time-Series-Decomposition-and-Hybrid-Model(CAL)
Implementation of the 2022 paper: https://doi.org/10.3390/e24020146

For this implementation, I used Python, most notably the keras and numpy libraries.

The procedure is as follows:
1. Import ticker data
2. Implement CEEMAD for extracting IMFs from the time series.
3. ADF test to determine if the IMF signals are stationary.
4. Stationary signals are fed into an ARMA(p,q) model.
5. Volatile/non-stationary signals are fed into LSTM model.
6. Sum the predictions of each resulting model to find Y(t+1)

I am left with a few questions where the paper is vague:

1. Normalzing data for LSTM? Do you denormalize before or after ensembling the LSTM predictions?
2. Validating Data?
3. Seasonality in the ARMA portion of the ensemble?
