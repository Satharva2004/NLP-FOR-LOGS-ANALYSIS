import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio

# Step 1: Generate Sample Data
# def generate_sample_data(num_days=366):
#     date_rng = pd.date_range(start="1/1/2024", end="12/31/2024", freq="H")
#     event_types = ["INFO", "ERROR", "WARNING", "LOGIN"]
#     user_ids = [f"user_{i}" for i in range(1, 101)]
#     ip_addresses = [f"192.168.0.{i}" for i in range(1, 101)]

#     data = {
#         "timestamp": date_rng,
#         "event": np.random.choice(event_types, size=len(date_rng)),
#         "user_id": np.random.choice(user_ids, size=len(date_rng)),
#         "ip_address": np.random.choice(ip_addresses, size=len(date_rng)),
#     }

#     df = pd.DataFrame(data)
#     df.to_csv("sample_logs.csv", index=False)
#     return df

# df = generate_sample_data()
# print(df.head())
# pio.renderers.default = 'browser'
# pio.renderers['browser'].port = 8086

file_path = r"C:\Users\sawan\OneDrive\Desktop\Python\NLP for Log\sample_logs.csv"
df = pd.read_csv(file_path, parse_dates=["timestamp"])


def plot_event_trend_interactive(df, event_type):
    df["date"] = df["timestamp"].dt.date
    trend_data = (
        df[df["event"] == event_type].groupby("date").size().reset_index(name="count")
    )

    fig = go.Figure()

    # Add a scatter plot for points
    fig.add_trace(
        go.Scatter(
            x=trend_data["date"],
            y=trend_data["count"],
            mode="markers",
            marker=dict(color="orange", size=8, opacity=0.5),
            hovertemplate="<b>Date</b>: %{x} <br><b>Count</b>: %{y}",
            name="Points",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=trend_data["date"],
            y=trend_data["count"],
            mode="lines",
            line=dict(color="blue", width=2),
            hovertemplate="<b>Date</b>: %{x} <br><b>Count</b>: %{y}",
            name="Trend",
        )
    )

    fig.update_layout(
        title=f"{event_type} Frequency Over Time",
        xaxis_title="Date",
        yaxis_title="Frequency",
        hovermode="closest",
        showlegend=True,
        template="none",
    )

    fig.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {"type": "scatter", "mode": "lines", "visible": True},
                        ],
                        "label": "Trend",
                        "method": "update",
                    },
                    {
                        "args": [
                            None,
                            {"type": "scatter", "mode": "markers", "visible": True},
                        ],
                        "label": "Points",
                        "method": "update",
                    },
                ],
                "direction": "down",
                "showactive": True,
            }
        ]
    )

    fig.show()


plot_event_trend_interactive(df, "INFO")
plot_event_trend_interactive(df, "ERROR")
plot_event_trend_interactive(df, "WARNING")
plot_event_trend_interactive(df, "LOGIN")
