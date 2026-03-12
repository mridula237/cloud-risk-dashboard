import pandas as pd
from db import engine

def get_correlation_matrix():

    query = """
        SELECT asset_id, date, daily_return
        FROM returns
    """

    df = pd.read_sql(query, engine)

    # ensure lowercase column names
    df.columns = df.columns.str.lower()

    # pivot into matrix form
    pivot = df.pivot(index="date", columns="asset_id", values="daily_return")

    # compute correlation
    corr = pivot.corr()

    # convert to long format for API
    corr = corr.reset_index().melt(
        id_vars="asset_id",
        var_name="asset2",
        value_name="correlation"
    )

    corr.rename(columns={"asset_id": "asset1"}, inplace=True)

    return corr.to_dict(orient="records")