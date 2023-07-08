import yfinance as yf
import pandas as pd
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score


symbols=sys.argv[1:]
for symbol in symbols:
    print(symbol[1:])
    path="./data/{}.csv".format(symbol[1:])
    if os.path.exists(path):
        ticket = pd.read_csv(f"{path}", index_col=0)

    else:
        ticket = yf.Ticker(f"{symbol}")
        ticket = ticket.history(period="max")
        ticket.to_csv(path)

    ticket.index = pd.to_datetime(ticket.index)

    try:
        del ticket["Dividends"]
        del ticket["Stock Splits"]

    except:
        pass
    ticket["next"] = ticket["Close"].shift(-1)
    ticket["Target"] = (ticket["next"] > ticket["Close"]).astype(int)

    ticket = ticket.loc["1990-01-01":].copy()
    print
    ticket.dropna(inplace=True)
    print(ticket.shape)


    # ticket.plot.line(y="Close", use_index=True)

    model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
    train = ticket.iloc[:-100]
    test = ticket.iloc[-100:]

    predictors = ["Close", "Volume", "Open", "High", "Low"]
    model.fit(train[predictors], train["Target"])

    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index)
    precision_score(test["Target"], preds)

    combined = pd.concat([test["Target"], preds], axis=1)


    def predict(train, test, predictors, model):
        model.fit(train[predictors], train["Target"])
        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index, name="Predictions")
        combined = pd.concat([test["Target"], preds], axis=1)
        return combined


    def backtest(data, model, predictors, start=2500, step=250):
        all_predictions = []

        for i in range(start, data.shape[0], step):
            train = data.iloc[0:i].copy()
            test = data.iloc[i:(i+step)].copy()
            predictions = predict(train, test, predictors, model)
            all_predictions.append(predictions)
        
        return pd.concat(all_predictions)

    predictions = backtest(ticket, model, predictors)
    predictions["Predictions"].value_counts()
    print('--------------------------------')
    print(precision_score(predictions["Target"], predictions["Predictions"]))
    print('--------------------------------')

    predictions["Target"].value_counts() / predictions.shape[0]
    horizons = [2,5,60,250,1000]
    new_predictors = []

    for horizon in horizons:
        rolling_averages = ticket.rolling(horizon).mean()
        
        ratio_column = f"Close_Ratio_{horizon}"
        ticket[ratio_column] = ticket["Close"] / rolling_averages["Close"]
        
        trend_column = f"Trend_{horizon}"
        ticket[trend_column] = ticket.shift(1).rolling(horizon).sum()["Target"]
        
        new_predictors+= [ratio_column, trend_column]

    ticket = ticket.dropna(subset=ticket.columns[ticket.columns != "next"])

    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)


    def predict(train, test, predictors, model):
        model.fit(train[predictors], train["Target"])
        preds = model.predict_proba(test[predictors])[:,1]
        preds[preds >=.6] = 1
        preds[preds <.6] = 0
        preds = pd.Series(preds, index=test.index, name="Predictions")
        combined = pd.concat([test["Target"], preds], axis=1)
        return combined

    predictions = backtest(ticket, model, new_predictors)


    predictions["Predictions"].value_counts()
    print("----------------")
    print(precision_score(predictions["Target"], predictions["Predictions"]))
    print("----------------")
    predictions["Target"].value_counts() / predictions.shape[0]

# predictions
