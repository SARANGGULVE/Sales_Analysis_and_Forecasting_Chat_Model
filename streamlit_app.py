"""
Orlin International — Sales Q&A Agent : Streamlit frontend
===========================================================
A chat-style web UI on top of the BigQuery + Ollama agent in test.py.

The agent fetches the BigQuery data once into a single DataFrame and answers
every question against that in-memory frame. This UI mirrors that: clients +
data are built once per session (cached), then reused.

Run with:
    streamlit run streamlit_app.py
"""

import pandas as pd
import streamlit as st

import test as agent  # reuse all the BigQuery / Ollama / chart logic


# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Orlin Sales Q&A Agent",
    page_icon="📊",
    layout="wide",
)


# ──────────────────────────────────────────────────────────────────────────────
# Cached clients + data  (built once per session)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Connecting to BigQuery and Ollama…")
def get_clients():
    """Initialise the BigQuery + Ollama clients exactly once."""
    return agent.init()


@st.cache_data(show_spinner="Fetching data from BigQuery…")
def get_data():
    """Fetch the BigQuery DataFrame once per session (Streamlit in-memory cache)."""
    bq, _ = get_clients()
    return agent.load_data(bq)


# ──────────────────────────────────────────────────────────────────────────────
# Core: fetch context + build chart for an intent (mirrors agent.run, but
# returns the Plotly figure instead of calling fig.show()).
# ──────────────────────────────────────────────────────────────────────────────

def context_and_chart(df, intent_key: str, show_chart: bool, forecast_sku=None, question=""):
    """Returns (context_text, plotly_figure_or_None, table_dataframe_or_None)."""
    fig = ctx = table = None

    if intent_key == "forecast":
        res = agent.forecast(df, sku=forecast_sku, horizon=agent.FORECAST_DAYS)
        fdf, m = res["forecast"], res["metrics"]
        table = fdf[["order_date", "revenue", "orders", "revenue_lo", "revenue_hi"]]
        ctx = (
            f"LightGBM {agent.FORECAST_DAYS}-Day Forecast — {res['label']} | "
            f"MAE ₹{m['mae']:,.0f} RMSE ₹{m['rmse']:,.0f}\n\n"
            + fdf[["order_date", "revenue", "orders"]].to_string(index=False)
        )
        if show_chart:
            fig = agent.chart_forecast(res)

    elif intent_key == "summary":
        d = agent._summary(df)
        table = pd.DataFrame([d])
        ctx = f"Sales Summary:\n{d}"
        if show_chart:
            fig = agent.kpi(d)

    elif intent_key == "last_month":
        d = agent._last_month(df)
        table = pd.DataFrame([d])
        ctx = f"Last Month ({d['period']}):\n{d}"
        if show_chart:
            fig = agent.kpi(d, f"📅 Last Month ({d['period']})")

    elif intent_key == "top_skus":
        by = agent.sku_rank_metric(question)
        agg = agent._top_skus(df, by=by)
        table = agg
        ctx = f"Top 20 SKUs by {by}:\n{agg.to_string(index=False)}"
        if show_chart:
            fig = agent.chart_skus(agg, by=by)

    elif intent_key == "channel":
        agg = agent._channel(df)
        table = agg
        ctx = f"Channel:\n{agg.to_string(index=False)}"
        if show_chart:
            fig = agent.chart_channel(agg)

    elif intent_key == "state":
        agg = agent._by_state(df)
        table = agg
        ctx = f"States:\n{agg.to_string(index=False)}"
        if show_chart:
            fig = agent.chart_state(agg)

    elif intent_key == "trend":
        dly = agent._daily(df)
        mo = (
            dly.groupby(dly["order_date"].dt.to_period("M"))
            .agg(orders=("orders", "sum"), revenue=("revenue", "sum"))
            .reset_index()
        )
        mo["order_date"] = mo["order_date"].astype(str)
        table = mo
        ctx = f"Monthly Trend:\n{mo.to_string(index=False)}"
        if show_chart:
            fig = agent.chart_trend(dly)

    elif intent_key == "status":
        agg = agent._status(df)
        table = agg
        ctx = f"Status:\n{agg.to_string(index=False)}"
        if show_chart:
            fig = agent.chart_status(agg)

    elif intent_key == "warehouse":
        agg = agent._warehouse(df)
        table = agg
        ctx = f"Warehouses:\n{agg.to_string(index=False)}"
        if show_chart:
            fig = agent.chart_warehouse(agg)

    elif intent_key == "discount":
        agg = agent._discount(df)
        table = agg
        ctx = f"Discount Analysis:\n{agg.to_string(index=False)}"
        if show_chart:
            fig = agent.chart_discount(agg)

    else:
        table = df.head(200)
        ctx = f"Sample ({len(df)} rows):\n{df.describe(include='all').to_string()}"

    return ctx, fig, table


# ──────────────────────────────────────────────────────────────────────────────
# Core: answer one question (intent → context/chart → LLM)
# ──────────────────────────────────────────────────────────────────────────────

def answer_question(df, llm, question: str, show_chart: bool, forecast_sku=None):
    """Returns (answer_text, plotly_figure_or_None, table_or_None)."""
    intent_key = agent.intent(question)
    ctx, fig, table = context_and_chart(df, intent_key, show_chart, forecast_sku, question)
    # Only surface the table when the user explicitly asks for tabular output.
    table = table if agent.wants_table(question) else None
    answer = llm.invoke(
        f"{agent.SYS}\n---\nQuestion: {question}\nData:\n{ctx}\n---\nAnswer:"
    )
    return answer, fig, table


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 Orlin Sales Agent")
    st.caption("BigQuery + Ollama powered Q&A")

    show_chart = st.toggle("Show charts", value=True)

    st.divider()
    st.subheader("💡 Example questions")
    example_clicked = None
    for q in agent.EXAMPLES:
        if st.button(q, use_container_width=True, key=f"ex_{q}"):
            example_clicked = q

    st.divider()
    if st.button("🔄 Refresh data", use_container_width=True,
                 help="Refetch from BigQuery"):
        get_data.clear()         # drop Streamlit's in-memory cache
        st.rerun()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# Connect + load data
# ──────────────────────────────────────────────────────────────────────────────

try:
    bq_client, llm_client = get_clients()
    data = get_data()
except Exception as e:
    st.error(f"❌ Startup error: {e}")
    st.info("Ensure GCP credentials are set in `.env` and Ollama is running.")
    st.stop()

st.success(
    f"Connected · BigQuery project **{bq_client.project}** · "
    f"LLM **{agent.OLLAMA_MODEL}** · **{len(data):,}** rows",
    icon="✅",
)


# ──────────────────────────────────────────────────────────────────────────────
# Forecast scope selector  (needs loaded data to list the top-20 SKUs)
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.divider()
    st.subheader("📈 Forecast scope")
    sku_options = ["Overall (all SKUs)"] + agent.top_sku_codes(data, 20)
    forecast_choice = st.selectbox(
        "SKU to forecast", sku_options,
        help="Top 20 SKUs by revenue. Applies whenever you ask for a forecast.",
    )
    forecast_sku = None if forecast_choice.startswith("Overall") else forecast_choice
    run_forecast = st.button("🔮 Run forecast", use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Chat history
# ──────────────────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay history (charts/tables are re-rendered from stored objects).
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("table") is not None:
            st.dataframe(msg["table"], use_container_width=True, key=f"tbl_{i}")
        if msg.get("fig") is not None:
            st.plotly_chart(msg["fig"], use_container_width=True, key=f"hist_{i}")


# ──────────────────────────────────────────────────────────────────────────────
# Handle input  (chat box or an example button)
# ──────────────────────────────────────────────────────────────────────────────

prompt = st.chat_input("Ask about sales, SKUs, channels, trends…")
forecast_question = (
    f"Forecast next {agent.FORECAST_DAYS} days for {forecast_choice}"
    if run_forecast else None
)
question = prompt or example_clicked or forecast_question

if question:
    # Show the user's message.
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate the answer.
    with st.chat_message("assistant"):
        with st.spinner("⏳ Analyzing data and thinking…"):
            try:
                answer, fig, table = answer_question(
                    data, llm_client, question, show_chart, forecast_sku
                )
            except Exception as e:
                answer, fig, table = f"❌ Error: {e}", None, None

        st.markdown(answer)
        if table is not None:
            st.dataframe(table, use_container_width=True)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True,
                            key=f"live_{len(st.session_state.messages)}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "fig": fig, "table": table}
    )
