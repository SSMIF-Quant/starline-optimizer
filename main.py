import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv

from starline_optimizer import OptimizationEngine

app = Flask(__name__)


@app.route("/", methods=["POST"])
def optimize():
    try:
        op = OptimizationEngine(request.get_json().get("tickers", []))

        u, t, shares_traded = op.execute(op._cash_only())[0]

        return jsonify({"weights": u, "time": t, "shares_traded": shares_traded})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_dotenv()
    app.run(port=int(os.environ.get("PORT", 8080)), debug=True)
