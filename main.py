import os
import traceback
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from pydantic import TypeAdapter, ValidationError
import numpy as np
import pandas as pd

from starline_optimizer import OptimizationEngine, TradeResult, logger

TRADE_PERIODS = 252
app = Flask(__name__)


def validate_request(req):
    """ Returns request params on success.
    On failure, throws an exception, which should be caught.
    The exception error message can be passed directly into the server response.

    :param req: Request body

    :return: Request params tickers, returns, varcovar
    """
    def validate_obj(obj: any, m: TypeAdapter, emsg: str):
        """ Wraps pydantic validation around try-catch so it throws an error with
        a predefined message.

        :param obj: Object to validate
        :param m: TypeAdapter to validate obj against
        :param emsg: Error message on failure

        :return: If validation succeeds, returns the validated object
        """
        try:
            return m.validate_python(obj), None
        except ValidationError:
            raise Exception(emsg)

    # Check request body params
    tickers = req.get("tickers", None)
    returns = req.get("returns", None)
    varcovar = req.get("varcovar", None)

    # Ensure required params exist
    if tickers is None:
        raise Exception("Request parameter 'tickers' required but not found.")
    if returns is None:
        raise Exception("Request parameter 'returns' required but not found.")
    if varcovar is None:
        raise Exception("Request parameter 'varcovar' required but not found.")

    # Check params types
    str_list = TypeAdapter(list[str])
    float_list = TypeAdapter(list[float])
    tickers, emsg = validate_obj(tickers, str_list, "Parameter 'tickers' must have type string[]")
    returns, emsg = validate_obj(returns, float_list, "Parameter 'returns' must have type float[]")
    varcovar, emsg = validate_obj(varcovar, float_list, "Parameter 'varcovar' must have type float[][]")

    # Check length constraints
    n = len(tickers)
    if n == 0:
        raise Exception("Parameter 'tickers' must not be empty")
    if len(returns[0]) != n or len(returns[1]) != n:
        raise Exception("Parameter 'returns' length doesn't match length of param 'tickers'")
    if len(varcovar) != n:
        raise Exception("Parameter 'varcovar' length doesn't match length of param 'tickers'")
    for vc_vec in varcovar:
        if len(vc_vec) != n:
            raise Exception("Parameter 'varcovar' is not square (length != width)")
    return tickers, returns, varcovar


@app.route("/", methods=["POST"])
def optimize():
    """Takes a list of tickers and optimizes a portfolio over them.

    :param tickers: Tickers to optimize a portfolio over.
    :param returns: 2 vectors of return forecasts for n tickers
                    The first vector represents returns now,
                    the second represents returns for an arbitrary next period
    :param varcovar: nxn variance covariance matrix

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

    logger.info(f"{request.host} received request from {request.origin}")
    body = request.get_json()

    try:
        tickers, returns, varcovar = validate_request(body)
    except Exception as e:
        reason = str(e)
        logger.error(f"{request.host} / POST failed: {reason}")
        return jsonify({"error": reason}), 400

    # Exec optimizer
    try:
        op = OptimizationEngine(tickers)

        starting_h = pd.Series(body.get("starting_portfolio", op._cash_only()))
        starting_h.index = np.append(op.data.tickers, "USDOLLAR")

        r_target = body.get("returns_target", None)
        sig_thresh = body.get("risk_threshold", None)

        trades = op.execute(starting_h, r_target=r_target, sig_thresh=sig_thresh)
        return jsonify(list(map(lambda t: trade_to_json(t, starting_h, op), trades)))
    except Exception as e:
        logger.error(f"{request.host} failed:\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_dotenv()
    app.run(port=int(os.environ.get("PORT", 8080)), debug=True)
