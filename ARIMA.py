import yfinance
import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
#Optimizaiton attempt
def ARIMA_OPT(data,max_p=30,max_d=4,max_q=30, train_coeff=.95):
    least_list = [(1,1,1),100]
    train = data[:int(len(data)*train_coeff)]
    test = data[int(len(data)*train_coeff):]
    start = len(train)
    end = len(train)+len(test)-1
    for p in range(max_p):
        for q in range(max_q):
            for d in range(max_d):
                #Model Creation
                arima_model = ARIMA(train,order=(p,d,q)).fit()
                pred = arima_model.predict(start=start, end=end, typ='levels')
                #Prediction mean
                forecast_mean = pred.mean()
                #Levelling prediction and test indexes
                pred.index = test.index
                mape1 = 100 * np.mean(np.abs((test - forecast_mean) / test))
                print(f"ARIMA({p},{d},{q}) MAPE:", mape1)
                if mape1<least_list[1]:
                    least_list = [(p,d,q),mape1]
    return least_list

#Variables
beta = .9  # train data size multiplier
file = "nikkei.csv"
col = "Adj Close"
freq = "D"
#Getting Data
nikkei = pd.read_csv(file,index_col="Date",parse_dates=True)
nik_adj = nikkei[col]
nik_adj = nik_adj.asfreq(freq).fillna(method="ffill")
train = nik_adj[:int(len(nik_adj)*beta)]
test = nik_adj[int(len(nik_adj)*beta):]
adfuller(nik_adj)[1]
adfuller(nik_adj.diff().dropna())[1]

#Auto ARIMA Suggestion 
model_fit = pm.arima.auto_arima(nik_adj, trace=True, suppress_warnings=True,seasonal_test=True,)
model_fit.summary()

#Seasonal Decompose
res_decompose = seasonal_decompose(nik_adj, period=365)
res_decompose_diff_1 = seasonal_decompose(nik_adj.diff().dropna(), period=365)

# Plotting the Decomposition Results
fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(9, 6), sharex=True)

# Original Series Decomposition
res_decompose.observed.plot(ax=axs[0, 0])
axs[0, 0].set_title('Original Series')
res_decompose.trend.plot(ax=axs[1, 0])
axs[1, 0].set_title('Trend')
res_decompose.seasonal.plot(ax=axs[2, 0])
axs[2, 0].set_title('Seasonality')
res_decompose.resid.plot(ax=axs[3, 0])
axs[3, 0].set_title('Residuals')

# First Order Differentiated Series Decomposition
res_decompose_diff_1.observed.plot(ax=axs[0, 1])
axs[0, 1].set_title('First Order Differentiated Series')
res_decompose_diff_1.trend.plot(ax=axs[1, 1])
axs[1, 1].set_title('Trend')
res_decompose_diff_1.seasonal.plot(ax=axs[2, 1])
axs[2, 1].set_title('Seasonality')
res_decompose_diff_1.resid.plot(ax=axs[3, 1])
axs[3, 1].set_title('Residuals')

fig.suptitle('Decomposition of Original vs. First Order Differentiated Series', fontsize=14)
fig.tight_layout()


#Multi Modelling list declarations
pdq_list = [(1,1,0),(0,1,1),(1,1,1),(2,1,2),(2,1,0),(0,1,2)]
result_list = []

#Multi Modelling
for order in pdq_list:
    #Model Creation
    arima_model = ARIMA(train,order=order)
    fit_arima = arima_model.fit()    
    fit_arima.summary()
    
    #Prediction Boundaries
    start = len(train)
    end = len(train)+len(test)-1
    
    #Prediction pd.Series
    pred = fit_arima.predict(start=start, end=end, typ='levels')
    
    #Prediction mean
    forecast_mean = pred.mean()
    
    #Levelling prediction and test indexes
    pred.index = test.index
    
    #Printing error statistics
    mae1 = np.mean(np.abs(test - forecast_mean))
    mape1 = 100 * np.mean(np.abs((test - forecast_mean) / test))
    rmse1 = np.sqrt(np.mean((test - forecast_mean)**2))
    result_list.append(pred)
    print("MAE:", mae1, f"--- ARIMA{order}")
    print("MAPE:", mape1, f"--- ARIMA{order}")
    print("RMSE:", rmse1, f"--- ARIMA{order}")



#Multi-Plotting
zoom = int(1450*len(train)/1500)
fig = plt.figure(figsize=(36, 24))

ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(234)
ax5 = fig.add_subplot(235)
ax6 = fig.add_subplot(236)

ax1.title.set_text(f'ARIMA {pdq_list[0]}')
ax2.title.set_text(f'ARIMA {pdq_list[1]}')
ax3.title.set_text(f'ARIMA {pdq_list[2]}')
ax4.title.set_text(f'ARIMA {pdq_list[3]}')
ax5.title.set_text(f'ARIMA {pdq_list[4]}')
ax6.title.set_text(f'ARIMA {pdq_list[5]}')

ax1.plot(train[zoom:], color="blue")
ax1.plot(test, color="orange")
ax1.plot(result_list[0], color="green")

ax2.plot(train[zoom:], color="blue")
ax2.plot(test, color="orange")
ax2.plot(result_list[1], color="green")

ax3.plot(train[zoom:], color="blue")
ax3.plot(test, color="orange")
ax3.plot(result_list[2], color="green")

ax4.plot(train[zoom:], color="blue")
ax4.plot(test, color="orange")
ax4.plot(result_list[3], color="green")

ax5.plot(train[zoom:], color="blue")
ax5.plot(test, color="orange")
ax5.plot(result_list[4], color="green")

ax6.plot(train[zoom:], color="blue")
ax6.plot(test, color="orange")
ax6.plot(result_list[5], color="green")

plt.show()

#Seasonal attempt
mod = sm.tsa.arima.ARIMA(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
res = mod.fit(method='innovations_mle', low_memory=True, cov_type='none')
start = len(train)
end = len(train)+len(test)-1
forecast1 = res.predict(start=start, end=end, typ='levels')
forecast1.index = test.index
plt.plot(train,color="blue", linewidth=3,linestyle="-")
plt.plot(test,color="orange", linewidth=3,linestyle="-")
plt.plot(forecast1,color="red", linewidth=1,linestyle="-")
plt.show()

