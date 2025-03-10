import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import numpy as np
import pandas as pd

from starline_optimizer import OptimizationEngine, TradeResult

TRADE_PERIODS = 252
app = Flask(__name__)


@app.route("/", methods=["POST"])
def optimize():
    """Takes a list of tickers and optimizes a portfolio over them.

    :param tickers: Tickers to optimize a portfolio over.

    Optional
    :param starting_portfolio: Cash value of each asset in the portfolio to start with
                               Asset value order must match ticker order
                               Must have one more element than the array of tickers
                               The extra last element represents cash position
                               Defaults to $1M cash with no assets
    :param returns_target: Annualized returns value to target
    :param risk_threshold: Annualized risk level to stay under

    :return: 70 potential portfolios
    """
    def trade_to_json(trade: TradeResult, starting_h: pd.Series, op: OptimizationEngine):
        tickers = op.data.tickers
        u, t, shares_traded = trade
        return {
                "trade": dict(zip(tickers + ["USDOLLAR"], u.to_list())),
                "exec_time": t,
                "shares_traded": dict(zip(tickers, shares_traded.to_list())),
                "annualized_return": op.h_return(starting_h + u) ** TRADE_PERIODS,
                "annualized_risk": op.h_risk((starting_h + u).iloc[:-1]) * TRADE_PERIODS
                }
    try:
        body = request.get_json()
        tickers = body["tickers"]

        op = OptimizationEngine(tickers)

        starting_h = pd.Series(body.get("starting_portfolio", op._cash_only()))
        starting_h.index = np.append(op.data.tickers, "USDOLLAR")

        r_target = body.get("returns_target", None)
        sig_thresh = body.get("risk_threshold", None)

        trades = op.execute(starting_h, r_target=r_target, sig_thresh=sig_thresh)
        return jsonify(list(map(lambda t: trade_to_json(t, starting_h, op), trades)))
    except KeyError:
        return jsonify({"error": "Request parameter 'ticker' required but not found."}), 400
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_dotenv()
    app.run(port=int(os.environ.get("PORT", 8080)), debug=True)
