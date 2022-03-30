# Stock-Index-Prediction-Based-on-Time-Series-Decomposition-and-Hybrid-Model(CAL)
Implementation of the 2022 paper: https://doi.org/10.3390/e24020146

For this implementation, I used Python, most notably the keras, PyEMD, pmdarima, and numpy libraries.

The procedure is as follows:
1. Import ticker data
2. Implement CEEMDAN for extracting intrinsic mode functions (IMFs) from the time series.
3. ADF test to determine if the IMF signals are stationary.
4. Stationary signals are fed into an ARMA(p,q) model.
5. Volatile/non-stationary signals are fed into LSTM model.
6. Sum the predictions of each resulting model to find Y(t+1)

I am left with a few questions/notes:

1. Normalzing data for LSTM? Do you denormalize before or after ensembling the LSTM predictions?
2. The authors note that 200 epochs results in the best model, but I found this to severly overfit. I use 50 epochs.
3. How are one-step-ahead predictions formed for the ARMA portion? Should we feed in the prior IMFs for each prediction, or just tell ARMA to generate n time steps into the future.
4. Seasonality in the ARMA portion of the ensemble... T or F?
