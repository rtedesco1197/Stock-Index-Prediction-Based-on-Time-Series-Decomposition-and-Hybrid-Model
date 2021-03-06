# Stock-Index-Prediction-Based-on-Time-Series-Decomposition-and-Hybrid-Model(CAL)
Implementation of the 2022 paper: https://doi.org/10.3390/e24020146
The main purpose of this project is to implement the CAL model as defined in the paper above as an ensemble of multiple LSTM and ARMA models which are fit depending on whether or not an intrinsic mode function (IMF) from CEEMDAN is stationary.

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
2. The authors note that 200 epochs results in the best model for predicting stock indexes, but I found this to severly overfit for the case of a single stock. I use 50 epochs to predict one day ahead for VOO.
3. How are one-step-ahead predictions formed for the ARMA portion? Should we feed in the prior IMFs for each prediction, or just tell ARMA to generate n time steps into the future (ie random walk it).
4. Seasonality in the ARMA portion of the ensemble... T or F?
5. I believe this model needs text/sentiment data before beginning to think about predictions further than one day ahead. 

Plots coming soon.
