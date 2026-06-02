"""
Orlin International — Sales Q&A Agent
BigQuery → Ollama LLM + Plotly charts + LightGBM 40-day forecast
Usage: python agent.py | python agent.py -q "..." | --no-chart
"""
import os, time, argparse, sys
from datetime import date, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
from google.cloud import bigquery
from langchain_ollama import OllamaLLM

# ── Config (loaded from .env) ──────────────────────────────────────────────────
load_dotenv()  # reads .env; GOOGLE_APPLICATION_CREDENTIALS is picked up by BigQuery

BQ_PROJECT    = os.getenv("GCP_PROJECT_ID", "orlinappareldataset")
BQ_DATASET    = os.getenv("BQ_DATASET", "oms_raw")
BQ_TABLE      = os.getenv("BQ_TABLE", "orders")
BQ_START      = os.getenv("BQ_START", "2025-04-01")
FULL_TABLE    = f"`{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}`"
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
FORECAST_DAYS = 40

T  = "plotly_dark"
PC, AC, SC, FC = "#00D4FF", "#FF6B6B", "#51CF66", "#FFD43B"
BG = "#0d1117"

# ── Init ──────────────────────────────────────────────────────────────────────
def init():
    creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    if creds and not os.path.exists(creds):
        raise FileNotFoundError(f"GOOGLE_APPLICATION_CREDENTIALS points to a missing file: {creds}")
    # With GOOGLE_APPLICATION_CREDENTIALS set, BigQuery uses that service-account
    # key; otherwise it falls back to Application Default Credentials (gcloud auth).
    bq = bigquery.Client(project=BQ_PROJECT or None)
    llm = OllamaLLM(model=OLLAMA_MODEL)
    auth = f"key:{os.path.basename(creds)}" if creds else "ADC"
    print(f"✓ BQ: {bq.project} ({auth})  ✓ LLM: {OLLAMA_MODEL}")
    return bq, llm

def bq_query(client, sql):
    # create_bqstorage_client=True streams results via the BigQuery Storage API
    # (Arrow + gRPC) instead of slow REST pagination — much faster downloads.
    return client.query(sql).to_dataframe(create_bqstorage_client=True)

# ── Data pipeline: BigQuery fetch ─────────────────────────────────────────────
def fetch_bq(client):
    print("📦 Fetching BigQuery data...")
    t0 = time.perf_counter()
    sql = f"""SELECT sku_code, listing_sku,
                     SAFE_CAST(qty AS INT64) qty, status, order_date,
                     SAFE_CAST(invoice_amount AS FLOAT64) invoice_amount,
                     SAFE_CAST(selling_price_per_item AS FLOAT64) selling_price_per_item,
                     SAFE_CAST(promo_discounts AS FLOAT64) promo_discounts,
                     SAFE_CAST(settlement_amount AS FLOAT64) settlement_amount,
                     buyer_city, buyer_state, channel, warehouse, company,
                     shipment_date, delivered_date, load_date
              FROM {FULL_TABLE}
              WHERE DATE(load_date) BETWEEN '{BQ_START}' AND CURRENT_DATE()"""
    df = bq_query(client, sql)
    # BigQuery returns nullable pandas dtypes (Int64/Float64); convert numeric
    # columns to plain numpy float64 so LightGBM / numpy ops accept them.
    for c in ["qty", "invoice_amount", "selling_price_per_item",
              "promo_discounts", "settlement_amount"]:
        if c in df.columns:
            df[c] = df[c].astype("float64")
    print(f"   ✅ {len(df)} rows in {time.perf_counter()-t0:.1f}s"); return df

def load_data(client):
    """Fetch the BigQuery orders data into a clean DataFrame."""
    return fetch_bq(client)

# ── Aggregation helpers (work on the loaded df) ───────────────────────────────
def _daily(df):
    df = df.copy(); df["order_date"] = pd.to_datetime(df["order_date"])
    g = df.groupby(df["order_date"].dt.date).agg(
        orders=("qty","count"), revenue=("invoice_amount","sum")).reset_index()
    g.columns = ["order_date","orders","revenue"]
    g["order_date"] = pd.to_datetime(g["order_date"]); return g

def _summary(df):
    return {"total_orders": len(df), "total_sales": df["invoice_amount"].sum(),
            "total_quantity": pd.to_numeric(df["qty"],errors="coerce").sum(),
            "total_skus": df["sku_code"].nunique()}

def _last_month(df):
    t = date.today().replace(day=1); e = t - timedelta(1); s = e.replace(day=1)
    mask = pd.to_datetime(df["order_date"]).dt.date.between(s, e)
    r = _summary(df[mask]); r["period"] = f"{s} → {e}"; return r

def _top_skus(df, n=20, by="revenue"):
    return (df.groupby("sku_code")
              .agg(revenue=("invoice_amount","sum"), total_qty=("qty","sum"), order_count=("sku_code","count"))
              .sort_values(by,ascending=False).head(n).reset_index())

def sku_rank_metric(q):
    """Decide whether 'top SKUs' means by quantity or by revenue."""
    ql = q.lower()
    if any(w in ql for w in ("quantity","qty","units","unit","volume","pieces","pcs")):
        return "total_qty"
    return "revenue"

def wants_table(q):
    """True if the user asked to see the data as a table."""
    ql = q.lower()
    return any(w in ql for w in ("tabular","table","grid","spreadsheet","as rows"))

def _channel(df):
    return (df.groupby("channel")
              .agg(orders=("sku_code","count"), revenue=("invoice_amount","sum"), total_qty=("qty","sum"))
              .sort_values("revenue",ascending=False).reset_index())

def _by_state(df):
    return (df.groupby("buyer_state")
              .agg(orders=("sku_code","count"), revenue=("invoice_amount","sum"))
              .sort_values("revenue",ascending=False).head(20).reset_index())

def _status(df):
    return (df.groupby("status")
              .agg(orders=("sku_code","count"), revenue=("invoice_amount","sum"))
              .sort_values("orders",ascending=False).reset_index())

def _warehouse(df):
    return (df.groupby("warehouse")
              .agg(orders=("sku_code","count"), revenue=("invoice_amount","sum"))
              .sort_values("revenue",ascending=False).reset_index())

def _discount(df):
    return (df.groupby("channel")
              .agg(avg_discount=("promo_discounts","mean"), avg_invoice=("invoice_amount","mean"),
                   avg_settlement=("settlement_amount","mean"), orders=("sku_code","count"))
              .sort_values("avg_discount",ascending=False).reset_index())

# ── LightGBM Forecast ─────────────────────────────────────────────────────────
def make_features(df):
    df = df.sort_values("order_date").set_index("order_date").asfreq("D").fillna(0).reset_index()
    d  = df["order_date"]
    df["dow"]=d.dt.dayofweek; df["dom"]=d.dt.day; df["doy"]=d.dt.dayofyear
    df["woy"]=d.dt.isocalendar().week.astype(int)
    df["month"]=d.dt.month; df["qtr"]=d.dt.quarter
    df["is_wknd"]=(d.dt.dayofweek>=5).astype(int); df["is_mend"]=d.dt.is_month_end.astype(int)
    for lag in [1,2,3,7,14,21,28]:
        df[f"rev_l{lag}"]=df["revenue"].shift(lag); df[f"ord_l{lag}"]=df["orders"].shift(lag)
    for w in [7,14,30]:
        df[f"rev_rm{w}"]=df["revenue"].shift(1).rolling(w).mean()
        df[f"rev_rs{w}"]=df["revenue"].shift(1).rolling(w).std()
        df[f"ord_rm{w}"]=df["orders"].shift(1).rolling(w).mean()
    df["rev_trend7"]=df["revenue"].shift(1).rolling(7).apply(
        lambda x: np.polyfit(range(7),x,1)[0] if len(x)==7 else 0, raw=True)
    return df.dropna()

def lgbm_forecast(df_merged, horizon=40, label="All SKUs"):
    import lightgbm as lgb
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    daily_df = _daily(df_merged)
    df   = make_features(daily_df.copy())
    if len(df) < 40:
        raise ValueError(f"Not enough history to forecast {label} "
                         f"({len(df)} usable days; need ≥ 40).")
    fc   = [c for c in df.columns if c not in ("order_date","revenue","orders")]
    X, yr, yo = df[fc].values, df["revenue"].values, df["orders"].values
    sp   = max(len(X)-30, int(len(X)*0.8))
    Xtr,Xv,ytr,yv,otr,ov = X[:sp],X[sp:],yr[:sp],yr[sp:],yo[:sp],yo[sp:]
    params = dict(objective="regression",metric="rmse",learning_rate=0.05,num_leaves=31,
                  n_estimators=500,min_child_samples=5,subsample=0.8,colsample_bytree=0.8,verbose=-1)
    def fit(yt,yv_):
        m = lgb.LGBMRegressor(**params)
        m.fit(Xtr,yt,eval_set=[(Xv,yv_)],callbacks=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(-1)])
        return m
    rm, om = fit(ytr,yv), fit(otr,ov)
    std  = np.std(yv - rm.predict(Xv))
    run  = daily_df.copy()
    dates = pd.date_range(df["order_date"].max()+timedelta(1), periods=horizon)
    revs,ords,los,his = [],[],[],[]
    for fd in dates:
        ft  = make_features(run); row = ft.iloc[[-1]][fc].values
        pr  = float(max(0,rm.predict(row)[0])); po = float(max(0,om.predict(row)[0]))
        revs.append(pr); ords.append(round(po))
        los.append(max(0,pr-1.96*std)); his.append(pr+1.96*std)
        run = pd.concat([run,pd.DataFrame({"order_date":[fd],"revenue":[pr],"orders":[round(po)]})],ignore_index=True)
    fcast = pd.DataFrame({"order_date":dates,"revenue":revs,"orders":ords,"revenue_lo":los,"revenue_hi":his})
    imp   = pd.DataFrame({"feature":fc,"importance":rm.feature_importances_}).sort_values("importance",ascending=False).head(15)
    mae   = mean_absolute_error(yv,rm.predict(Xv)); rmse = np.sqrt(mean_squared_error(yv,rm.predict(Xv)))
    return {"history":daily_df,"forecast":fcast,"importance":imp,"label":label,
            "metrics":{"mae":round(mae,2),"rmse":round(rmse,2),"train":sp,"val":len(Xv)}}

def top_sku_codes(df, n=20):
    """Return the top-n SKU codes by revenue (for forecast selection)."""
    return _top_skus(df, n)["sku_code"].astype(str).tolist()

def forecast(df, sku=None, horizon=FORECAST_DAYS):
    """Forecast revenue/orders overall (sku=None) or for a single SKU."""
    if sku:
        sub = df[df["sku_code"].astype(str) == str(sku)]
        if sub.empty:
            raise ValueError(f"No data found for SKU '{sku}'.")
        return lgbm_forecast(sub, horizon, label=f"SKU {sku}")
    return lgbm_forecast(df, horizon, label="All SKUs")

# ── Charts ────────────────────────────────────────────────────────────────────
def kpi(data, title="📊 Sales Summary"):
    labels,vals,prefixes,colors = (["Total Orders","Total Revenue","Total Qty","Unique SKUs"],
        [data.get("total_orders",0),data.get("total_sales",0),data.get("total_quantity",0),data.get("total_skus",0)],
        ["","₹","",""],[PC,AC,SC,"#FFA94D"])
    fig = make_subplots(rows=1,cols=4,specs=[[{"type":"indicator"}]*4])
    for i,(l,v,p,c) in enumerate(zip(labels,vals,prefixes,colors),1):
        fig.add_trace(go.Indicator(mode="number",value=v,title={"text":l,"font":{"size":13}},
            number={"prefix":p,"font":{"size":26,"color":c}}),row=1,col=i)
    fig.update_layout(template=T,title=title,height=240,paper_bgcolor=BG,plot_bgcolor=BG); return fig

def chart_forecast(res):
    label = res.get("label","All SKUs")
    h = res["history"].copy(); h["order_date"]=pd.to_datetime(h["order_date"])
    h = h[h["order_date"]>=h["order_date"].max()-timedelta(90)]
    f,imp,m = res["forecast"].copy(),res["importance"],res["metrics"]
    fig = make_subplots(rows=2,cols=2,row_heights=[0.65,0.35],column_widths=[0.72,0.28],
        specs=[[{"colspan":2,"secondary_y":True},None],[{"type":"bar"},{"type":"indicator"}]],
        subplot_titles=(f"📈 {label}: Revenue + {FORECAST_DAYS}-Day Forecast","","🔍 Feature Importance","📐 Accuracy"),
        vertical_spacing=0.12,horizontal_spacing=0.08)
    fig.add_trace(go.Scatter(x=h["order_date"],y=h["revenue"],name="Historical Revenue",
        fill="tozeroy",fillcolor="rgba(0,212,255,0.08)",line=dict(color=PC,width=2)),row=1,col=1,secondary_y=False)
    fig.add_trace(go.Scatter(x=pd.concat([f["order_date"],f["order_date"][::-1]]),
        y=pd.concat([f["revenue_hi"],f["revenue_lo"][::-1]]),fill="toself",
        fillcolor="rgba(255,212,59,0.15)",line=dict(color="rgba(0,0,0,0)"),name="95% Band"),row=1,col=1,secondary_y=False)
    fig.add_trace(go.Scatter(x=f["order_date"],y=f["revenue"],name="Forecast Revenue",
        line=dict(color=FC,width=2.5,dash="dash"),mode="lines+markers",marker=dict(size=4,color=FC)),row=1,col=1,secondary_y=False)
    fig.add_trace(go.Bar(x=h["order_date"],y=h["orders"],name="Hist Orders",
        marker_color="rgba(81,207,102,0.35)",width=86400000),row=1,col=1,secondary_y=True)
    fig.add_trace(go.Bar(x=f["order_date"],y=f["orders"],name="Fcst Orders",
        marker_color="rgba(255,107,107,0.45)",width=86400000),row=1,col=1,secondary_y=True)
    fig.add_vline(x=h["order_date"].max().timestamp()*1000,
        line=dict(color="rgba(255,255,255,0.3)",dash="dot",width=1), row=1, col=1)
    imp_s = imp.sort_values("importance")
    fig.add_trace(go.Bar(x=imp_s["importance"],y=imp_s["feature"],orientation="h",
        marker=dict(color=imp_s["importance"],colorscale=[[0,"#1a1a2e"],[1,FC]]),showlegend=False),row=2,col=1)
    fig.add_trace(go.Indicator(mode="number",value=m["mae"],title={"text":"MAE Revenue (₹)"},
        number={"prefix":"₹","font":{"size":22,"color":SC}}),row=2,col=2)
    fig.update_layout(template=T,height=700,paper_bgcolor=BG,plot_bgcolor=BG,barmode="overlay",
        legend=dict(orientation="h",yanchor="bottom",y=1.01,xanchor="right",x=1,bgcolor="rgba(0,0,0,0)"),
        title=f"<b>LightGBM Forecast — {label}</b> | MAE ₹{m['mae']:,.0f}  RMSE ₹{m['rmse']:,.0f}  Train:{m['train']} Val:{m['val']}")
    fig.update_yaxes(title_text="Revenue (₹)",secondary_y=False,row=1,col=1)
    fig.update_yaxes(title_text="Orders",secondary_y=True,row=1,col=1); return fig

def chart_skus(df, by="revenue"):
    df = df.sort_values(by)
    is_rev = by == "revenue"
    label  = "Revenue" if is_rev else "Quantity"
    text   = df[by].apply(lambda v: f"₹{v:,.0f}" if is_rev else f"{v:,.0f}")
    fig = px.bar(df,x=by,y="sku_code",orientation="h",color=by,
        color_continuous_scale=["#1a1a2e",PC],text=text,
        title=f"🏆 Top SKUs by {label}",template=T)
    fig.update_traces(textposition="outside")
    fig.update_layout(height=600,coloraxis_showscale=False,paper_bgcolor=BG,plot_bgcolor=BG); return fig

def chart_channel(df):
    fig = make_subplots(rows=1,cols=2,specs=[[{"type":"pie"},{"type":"bar"}]],
                        subplot_titles=("Revenue Share","Orders vs Revenue"))
    fig.add_trace(go.Pie(labels=df["channel"],values=df["revenue"],hole=0.45,
        marker_colors=px.colors.qualitative.Bold),row=1,col=1)
    fig.add_trace(go.Bar(x=df["channel"],y=df["orders"],name="Orders",marker_color=PC),row=1,col=2)
    fig.add_trace(go.Bar(x=df["channel"],y=df["revenue"],name="Revenue",marker_color=AC),row=1,col=2)
    fig.update_layout(template=T,title="📦 Channel Breakdown",barmode="group",height=430,paper_bgcolor=BG,plot_bgcolor=BG); return fig

def chart_state(df):
    fig = px.bar(df,x="buyer_state",y="revenue",color="revenue",color_continuous_scale=["#1a1a2e",SC],
        title="🗺️ Revenue by State",template=T)
    fig.update_layout(height=460,coloraxis_showscale=False,paper_bgcolor=BG,plot_bgcolor=BG,xaxis_tickangle=-40); return fig

def chart_trend(df):
    mo = df.groupby(df["order_date"].dt.to_period("M")).agg(orders=("orders","sum"),revenue=("revenue","sum")).reset_index()
    mo["order_date"] = mo["order_date"].dt.to_timestamp()
    fig = make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Scatter(x=mo["order_date"],y=mo["revenue"],name="Revenue",fill="tozeroy",line=dict(color=PC,width=2)),secondary_y=False)
    fig.add_trace(go.Bar(x=mo["order_date"],y=mo["orders"],name="Orders",marker_color=AC,opacity=0.6),secondary_y=True)
    fig.update_layout(template=T,title="📈 Monthly Trend",height=430,paper_bgcolor=BG,plot_bgcolor=BG); return fig

def chart_status(df):
    fig = make_subplots(rows=1,cols=2,specs=[[{"type":"pie"},{"type":"bar"}]],
                        subplot_titles=("Order Share","Revenue by Status"))
    fig.add_trace(go.Pie(labels=df["status"],values=df["orders"],hole=0.4),row=1,col=1)
    fig.add_trace(go.Bar(x=df["status"],y=df["revenue"],marker_color=px.colors.qualitative.Bold[:len(df)]),row=1,col=2)
    fig.update_layout(template=T,title="📋 Order Status",height=400,paper_bgcolor=BG,plot_bgcolor=BG); return fig

def chart_warehouse(df):
    fig = px.bar(df,x="warehouse",y=["orders","revenue"],barmode="group",
        color_discrete_sequence=[PC,AC],title="🏭 Warehouse Performance",template=T)
    fig.update_layout(height=400,paper_bgcolor=BG,plot_bgcolor=BG); return fig

def chart_discount(df):
    fig = go.Figure([go.Bar(x=df["channel"],y=df["avg_discount"],name="Avg Discount",marker_color=AC),
                     go.Bar(x=df["channel"],y=df["avg_invoice"],name="Avg Invoice",marker_color=PC),
                     go.Bar(x=df["channel"],y=df["avg_settlement"],name="Avg Settlement",marker_color=SC)])
    fig.update_layout(barmode="group",template=T,title="💸 Discount vs Invoice vs Settlement",height=400,paper_bgcolor=BG,plot_bgcolor=BG); return fig

# ── Intent ────────────────────────────────────────────────────────────────────
INTENT_MAP = {
    "forecast":   ["forecast","predict","next 40","future","projection","next month days"],
    "last_month": ["last month","previous month"],
    "top_skus":   ["top sku","best sku","best selling","top product","top selling"],
    "channel":    ["channel","platform","marketplace","amazon","flipkart","meesho"],
    "state":      ["state","city","region","location","geography"],
    "trend":      ["daily","trend","over time","monthly","weekly","period"],
    "status":     ["status","delivered","cancelled","returned","pending"],
    "warehouse":  ["warehouse","fulfillment"],
    "discount":   ["discount","promo","settlement","margin"],
    "summary":    ["total","overall","summary","revenue","sales","sale","how many order"],
}

# Ranking cues that, combined with "sku"/"product", mean "top SKUs"
_SKU_RANK_CUES = ["top","best","highest","most","rank","sell","sale","revenue","by "]

def intent(q):
    ql = q.lower()
    # SKU ranking questions ("top 10 SKU by sale", "best selling products", …).
    # Checked first so they aren't swallowed by the generic "revenue"/"sales".
    if ("sku" in ql or "product" in ql) and any(c in ql for c in _SKU_RANK_CUES):
        return "top_skus"
    for k,kws in INTENT_MAP.items():
        if any(kw in ql for kw in kws): return k
    return "sample"

# ── Dispatcher ────────────────────────────────────────────────────────────────
def run(df, intent_key, show_chart=True, question=""):
    """All analytics run on the already-loaded BigQuery DataFrame `df`."""
    fig = ctx = None
    if intent_key == "forecast":
        print("\n  🔮 Training LightGBM on merged data…", flush=True)
        res  = lgbm_forecast(df, FORECAST_DAYS)
        fdf  = res["forecast"]; m = res["metrics"]
        ctx  = (f"LightGBM {FORECAST_DAYS}-Day Forecast | MAE ₹{m['mae']:,.0f} RMSE ₹{m['rmse']:,.0f}\n\n"
                + fdf[["order_date","revenue","orders"]].to_string(index=False))
        if show_chart: fig = chart_forecast(res)
    elif intent_key == "summary":
        d = _summary(df); ctx = f"Sales Summary:\n{d}"
        if show_chart: fig = kpi(d)
    elif intent_key == "last_month":
        d = _last_month(df); ctx = f"Last Month ({d['period']}):\n{d}"
        if show_chart: fig = kpi(d, f"📅 Last Month ({d['period']})")
    elif intent_key == "top_skus":
        by = sku_rank_metric(question)
        agg = _top_skus(df, by=by); ctx = f"Top 20 SKUs by {by}:\n{agg.to_string(index=False)}"
        if show_chart: fig = chart_skus(agg, by=by)
    elif intent_key == "channel":
        agg = _channel(df); ctx = f"Channel:\n{agg.to_string(index=False)}"
        if show_chart: fig = chart_channel(agg)
    elif intent_key == "state":
        agg = _by_state(df); ctx = f"States:\n{agg.to_string(index=False)}"
        if show_chart: fig = chart_state(agg)
    elif intent_key == "trend":
        dly = _daily(df)
        mo  = dly.groupby(dly["order_date"].dt.to_period("M")).agg(orders=("orders","sum"),revenue=("revenue","sum")).reset_index()
        ctx = f"Monthly Trend:\n{mo.to_string(index=False)}"
        if show_chart: fig = chart_trend(dly)
    elif intent_key == "status":
        agg = _status(df); ctx = f"Status:\n{agg.to_string(index=False)}"
        if show_chart: fig = chart_status(agg)
    elif intent_key == "warehouse":
        agg = _warehouse(df); ctx = f"Warehouses:\n{agg.to_string(index=False)}"
        if show_chart: fig = chart_warehouse(agg)
    elif intent_key == "discount":
        agg = _discount(df); ctx = f"Discount Analysis:\n{agg.to_string(index=False)}"
        if show_chart: fig = chart_discount(agg)
    else:
        ctx = f"Sample ({len(df)} rows):\n{df.describe(include='all').to_string()}"
    if fig: fig.show()
    return ctx

# ── LLM ───────────────────────────────────────────────────────────────────────
SYS = """You are a senior Sales Analyst for Orlin International.
Data: sku_code, qty, status, order_date, invoice_amount, promo_discounts,
settlement_amount, buyer_city/state, channel, warehouse, shipment/delivered_date.
Use ONLY the data provided below. Never invent SKU codes, numbers, or rows that
are not present in the data; if the data does not answer the question, say so.
Answer concisely with bullets/tables. End with 1-2 actionable recommendations."""

def ask(df, llm, question, show_chart=True):
    ctx = run(df, intent(question), show_chart, question)
    return llm.invoke(f"{SYS}\n---\nQuestion: {question}\nData:\n{ctx}\n---\nAnswer:")

# ── CLI ───────────────────────────────────────────────────────────────────────
EXAMPLES = ["Overall sales summary","Last month sale","Forecast next 40 days",
            "Top 10 SKUs by revenue","Channel revenue breakdown","Revenue by state",
            "Monthly sales trend","Order status breakdown","Warehouse performance",
            "Average discount by channel"]

def loop(df, llm, show_chart=True):
    print(f"\n{'═'*58}\n  Orlin International — Sales Agent 🚀\n  {BQ_PROJECT} | {OLLAMA_MODEL}\n{'═'*58}")
    print("Commands: 'examples'  'quit'\n")
    while True:
        try: question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt): print("\nGoodbye!"); break
        if not question: continue
        if question.lower() in ("quit","exit","q"): print("Goodbye!"); break
        if question.lower() == "examples":
            print("\n" + "\n".join(f"  {i:>2}. {e}" for i,e in enumerate(EXAMPLES,1)) + "\n"); continue
        print("\n⏳ Thinking…", end="", flush=True)
        try: print(f"\r\n🤖 Agent:\n{ask(df, llm, question, show_chart)}\n")
        except Exception as e: print(f"\r\n❌ {e}\n")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--question","-q", type=str, default=None)
    p.add_argument("--no-chart", action="store_true")
    a = p.parse_args()
    try:
        bq, llm = init()
        df = load_data(bq)          # ← BigQuery fetch at startup
    except Exception as e: print(f"❌ {e}"); sys.exit(1)
    if a.question: print(f"\n🤖 {ask(df, llm, a.question, not a.no_chart)}")
    else: loop(df, llm, not a.no_chart)

if __name__ == "__main__": main()