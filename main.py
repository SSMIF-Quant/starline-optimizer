import os
import traceback
import uuid
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

    :return: Request params tickers, prices, returns, varcovar, r_target, sig_thresh
    """
    def validate_param(obj: any, m: TypeAdapter, emsg: str):
        """ Wraps pydantic validation around try-catch so it throws an error with
        a predefined message.

        :param obj: Object to validate
        :param m: TypeAdapter to validate obj against
        :param emsg: Error message on failure

        :return: If validation succeeds, returns the validated object
        """
        try:
            return m.validate_python(obj)
        except ValidationError:
            raise Exception(emsg)

    # Get request body params
    tickers = req.get("tickers", None)
    prices = req.get("prices", None)  # TODO currently implemented as a vector of floats
    returns = req.get("returns", None)
    varcovar = req.get("varcovar", None)
    r_target = req.get("returns_target", None)
    sig_thresh = req.get("risk_threshold", None)

    # Ensure required params exist
    if tickers is None:
        raise Exception("Request parameter 'tickers' required but not found.")
    if prices is None:
        raise Exception("Request parameter 'prices' required but not found.")
    if returns is None:
        raise Exception("Request parameter 'returns' required but not found.")
    if varcovar is None:
        raise Exception("Request parameter 'varcovar' required but not found.")

    # Check params types
    strl = TypeAdapter(list[str])
    floatl = TypeAdapter(list[float])
    floatll = TypeAdapter(list[list[float]])
    tickers = validate_param(tickers, strl, "Parameter 'tickers' must have type string[]")
    prices = validate_param(prices, floatl, "Parameter 'returns' must have type float[]")
    returns = validate_param(returns, floatll, "Parameter 'returns' must have type float[][2]")
    varcovar = validate_param(varcovar, floatll, "Parameter 'varcovar' must have type float[][]")
    if r_target is not None and not isinstance(r_target, float):
        raise Exception("Optional parameter 'r_target' must have type float.")
    if sig_thresh is not None and not isinstance(sig_thresh, float):
        raise Exception("Optional parameter 'sig_thresh' must have type float.")
    if sig_thresh is not None and sig_thresh <= 0:
        raise Exception("Optional parameter 'sig_thresh' must be a positive float.")

    # Check length constraints
    n = len(tickers)
    if n == 0:
        raise Exception("Parameter 'tickers' must not be empty")
    if len(returns) != 2:
        raise Exception("Parameter 'returns' must have exactly 2 expected returns vectors.")
    if len(prices) != n:
        raise Exception("Parameter 'prices' length doesn't match length of param 'tickers'")
    if len(returns[0]) != n:
        raise Exception("Parameter 'returns[0]' length doesn't match length of param 'tickers'")
    if len(returns[1]) != n:
        raise Exception("Parameter 'returns[1]' length doesn't match length of param 'tickers'")
    if len(varcovar) != n:
        raise Exception("Parameter 'varcovar' length doesn't match length of param 'tickers'")
    for vc_vec in varcovar:
        if len(vc_vec) != n:
            raise Exception("Parameter 'varcovar' is not square (length != width)")
    return tickers, prices, returns, varcovar, r_target, sig_thresh


@app.route("/", methods=["POST"])
def optimize():
    """Takes a list of tickers and optimizes a portfolio over them.

    :param tickers: Tickers to optimize a portfolio over.
    :param prices: n-length vector of current asset prices.
    :param returns: 2 vectors of return forecasts for n assets
                    The first vector represents forecasts for today
                    The second vector represents forecasts for the next trading period
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
                "trade_dollar_value": dict(zip(tickers + ["USDOLLAR"], u.to_list())),
                "exec_time": t,
                "shares_traded": dict(zip(tickers, shares_traded.to_list())),
                "annualized_return": op.h_return(starting_h + u) ** TRADE_PERIODS,
                "annualized_risk": op.h_risk((starting_h + u).iloc[:-1]) * TRADE_PERIODS
                }

    logger.info(f"{request.host} received request from {request.origin}")
    body = request.get_json()

    try:
        tickers, prices, returns, varcovar, r_target, sig_thresh = validate_request(body)
    except Exception as e:
        reason = str(e)
        logger.error(f"{request.host} / POST failed: {reason}")
        return jsonify({"error": reason}), 400

    job_uuid = uuid.uuid1()
    # TODO end request with success + job uuid response and run the job

    # Convert param vectors into DataFrame
    # TODO current returns forecasts timestamps defaults to today and one month after
    now = pd.Timestamp.today().floor("d")
    one_mo_after = now + pd.Timedelta(30, "days")
    prices = pd.Series(prices, index=tickers)
    returns = pd.DataFrame(returns, index=[now, one_mo_after], columns=tickers)
    varcovar = pd.DataFrame(varcovar, index=tickers, columns=tickers)

    # Exec optimizer
    try:
        op = OptimizationEngine(tickers, job_uuid, prices, returns, varcovar)

        starting_h = pd.Series(body.get("starting_portfolio", op._cash_only()))
        starting_h.index = np.append(op.data.tickers, "USDOLLAR")

        trades = op.execute(starting_h, r_target=r_target, sig_thresh=sig_thresh)
        logger.success(f"{request.host} / POST succeeded: Job {job_uuid} resolved")
        return jsonify(list(map(lambda t: trade_to_json(t, starting_h, op), trades)))
    except Exception as e:
        reason = f"{request.host} / POST failed: Job {job_uuid} failed:\n{traceback.format_exc()}"
        logger.error(reason)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_dotenv()
    app.run(port=int(os.environ.get("PORT", 8080)), debug=True)
