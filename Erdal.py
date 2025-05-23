import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import matplotlib.pyplot as plt
import io
from scipy.io import loadmat
from PIL import Image, ImageDraw
from folium.plugins import MarkerCluster
import configparser
import os
import configparser
import numpy as np

# Language selector
lang = st.sidebar.selectbox("Language / Språk", ["English", "Norwegian"])
lang_code = 'en' if lang == "English" else 'no'

# Load translation file
config = configparser.ConfigParser()
config.read(f'language_{lang_code}.ini')

_ = lambda key: config.get("texts", key, fallback=key)  # fallback if key missing


st.sidebar.title("Contents")
page = st.sidebar.radio("Select a page:", ["Erdal, Askøy", "Infiltration and Inflow", "Zone Map and Recommendation", "Steinrusten Rain Gauge", "Erdal Water Consumption", "Sensor and Zone Data"])

# Conditional logic for different pages
if page == "Erdal, Askøy":
    # Display image as header
    #st.image("Header.jpg", use_column_width=True)
    st.title("Infiltration and Inflow in Erdal, Askøy")
    st.write("To quantify the infiltration and inflow in Erdal, Askøy, several methods has been used.")
    st.write("To look at the quantification methods, zone map, and other relevant data from this research, choose a page from the side bar.")

    st.title("Erdal")

    # Define the coordinates for the map center
    latitude = 60.447644
    longitude = 5.2201019

    # Create a Folium map centered at the specified coordinates
    m = folium.Map(location=[latitude, longitude], zoom_start=15)
       # Display the map in Streamlit
    st_folium(m, width=700, height=500)
    st.subheader("Catchment Zones")
    st.image("ERDAL.png", caption="Zones", use_column_width=True)
    st.write("The map over illustrates the sewer zones in the Erdal.")

elif page == "Infiltration and Inflow":
    st.title("Quantification of Infiltration and inflow")
    st.write("This page show results from the different methods conducted. First a summary, then individually results with visualisations.")

     # Summary of the Results
    ii_data = [
        {"Zone": "Zone 1", "Sensor": "Sensor 1", "DWF I/I Status": "✅ Normal", "MNF I/I Status": "✅ Normal", "DWF vs WWT I/I Status": "🚨 High I/I", "SWMM I/I Status": "✅ Normal", "Status": "✅ Normal"},
        {"Zone": "Zone 2", "Sensor": "Sensor 2", "DWF I/I Status": "✅ Normal", "MNF I/I Status": "✅ Normal", "DWF vs WWT I/I Status": "🚨 High I/I", "SWMM I/I Status": "✅ Normal", "Status": "✅ Normal"},
        {"Zone": "Zone 3", "Sensor": "Sensor 3", "DWF I/I Status": "🚨 High I/I", "MNF I/I Status": "🚨 High I/I", "DWF vs WWT I/I Status": "🚨 High I/I", "SWMM I/I Status": "🚨 High I/I", "Status": "🚨 High I/I"},
        {"Zone": "Zone 4", "Sensor": "Sensor 4", "DWF I/I Status": "✅ Normal", "MNF I/I Status": "✅ Normal", "DWF vs WWT I/I Status": "🚨 High I/I", "SWMM I/I Status": "✅ Normal", "Status": "✅ Normal"},
        {"Zone": "Zone 6", "Sensor": "Sensor 6", "DWF I/I Status": "🚨 High I/I", "MNF I/I Status": "🚨 High I/I", "DWF vs WWT I/I Status": "🚨 High I/I", "SWMM I/I Status": "✅ Normal", "Status": "🟠 Potential I/I"},
        {"Zone": "Zone 7", "Sensor": "Sensor 7", "DWF I/I Status": "🚨 High I/I", "MNF I/I Status": "🚨 High I/I", "DWF vs WWT I/I Status": "🚨 High I/I", "SWMM I/I Status": "🚨 High I/I", "Status": "🚨 High I/I"},
        {"Zone": "Zone 8", "Sensor": "Sensor 8", "DWF I/I Status": "-", "MNF I/I Status": "🚨 High I/I", "DWF vs WWT I/I Status": "🚨 High I/I", "SWMM I/I Status": "🚨 High I/I", "Status": "🚨 High I/I"},
    ]

    # Convert to DataFrame
    df = pd.DataFrame(ii_data)

    # Format and style the table
    styled_df = df.style.applymap(
    lambda val: (
        'color: red; font-weight: bold' if "🚨" in str(val) else
        'color: green; font-weight: bold' if "✅" in str(val) else
        'color: orange; font-weight: bold' if "🟠" in str(val) else ''
    ),
    subset=["DWF I/I Status", "MNF I/I Status", "DWF vs WWT I/I Status", "SWMM I/I Status", "Status"]
    )   

    # Display it
    st.subheader("Summary")
    st.dataframe(styled_df)

        # Combined dataset
    df = pd.DataFrame({
        "Zone": ["1", "2", "3", "4", "6", "7", "8"],
        "DWF I/I %": [-850.2, -163.0, 61.3, -13.5, 42.7, 74.3, 43.2],
        "Simulated I/I %": [0.0, 2.6, 30.3, 2.7, 2.0, 66.0, 59.0],
        "MNF (m³/day)": [0.26, 1.73, 12.08, 2.04, 10.71, 52.48, 408.86]
    })

    # Normalize MNF for marker size scaling
    df["Size"] = df["MNF (m³/day)"] * 5  

    # Scatter plot
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(df["Simulated I/I %"], df["DWF I/I %"], s=df["Size"],
                         c=df["MNF (m³/day)"], cmap="Blues", edgecolor="black")

    # Add annotations
    for _, row in df.iterrows():
        ax.text(row["Simulated I/I %"] + 1, row["DWF I/I %"], f"Zone {row['Zone']}", fontsize=9)

    # Axes, labels, and colorbar
    ax.set_xlabel("Simulated I/I %")
    ax.set_ylabel("DWF-based I/I %")
    ax.set_title("I/I Comparison by Method (Zone Level)")
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.colorbar(scatter, label="MNF (m³/day)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    st.pyplot(fig)

    st.title("Dry Weather Flow Method (DWF)")
    st.markdown("""
    This method estimates **infiltration under dry conditions**, based on the difference between total measured flow and actual water consumption during dry weather.
    """)

    zones = [1, 2, 3, 4, 6, 7, 8]
    ii_results = []

    # Thresholds
    volume_threshold = 10000  # in L
    percent_threshold = 15    # %

    for zone in zones:
        try:
            # Load sensor-measured flow (in L/s)
            df_flow = pd.read_csv(f"Zone{zone}_DWF.csv", parse_dates=["Time"])
            if "FlowDifference" in df_flow.columns:
                df_flow.rename(columns={"FlowDifference": "Flow"}, inplace=True)
            df_flow.set_index("Time", inplace=True)

            total_flow = df_flow["Flow"].sum() * 60  # Convert to Liters per 5-min interval

            # Load water consumption (in m³/h)
            df_cons = pd.read_csv(f"DWF_WaterConsumption_Zone{zone}.csv", parse_dates=["Time"])
            df_cons.rename(columns={"consumption": "Consumption"}, inplace=True)
            df_cons.set_index("Time", inplace=True)

            # Convert m³/h → L per 5 min
            df_cons["Consumption_L"] = df_cons["Consumption"] * 1000 / 12

            if zone == 8:
                # Adjust total water consumption with pumping data
                df_cons["Consumption_L"] *= 1500289.8 / df_cons["Consumption_L"].sum()

            total_consumption = df_cons["Consumption_L"].sum()

            # Calculate I/I
            ii_volume = total_flow - total_consumption
            ii_percent = (ii_volume / total_flow) * 100 if total_flow > 0 else 0

            # Flag
            status = "🚨 High I/I" if ii_volume > volume_threshold or ii_percent > percent_threshold else "✅ Normal"

            ii_results.append({
                "Zone": f"Zone {zone}",
                "Measured Flow (L)": round(total_flow, 2),
                "Water Consumption (L)": round(total_consumption, 2),
                "Estimated Infiltration (L)": round(ii_volume, 2),
                "Infiltration % of Flow": round(ii_percent, 1),
                "Status": status
            })

        except Exception as e:
            st.warning(f"Zone {zone}: Error calculating I/I – {e}")

    # Display
    df_result = pd.DataFrame(ii_results)
    styled_df = df_result.style.applymap(
        lambda val: 'color: red; font-weight: bold' if val == "🚨 High I/I" else '',
        subset=["Status"]
    )
    st.dataframe(styled_df)
    st.markdown("### Estimated Dry Weather Infiltration by Zone")
    #st.bar_chart(df_result.set_index("Zone")["Estimated Infiltration (L)"])

    zones = [1, 2, 3, 4, 6, 7, 8]
    selected_zone = st.selectbox("Choose a zone:", zones)

    try:
        # Load Measured Flow
        df_flow = pd.read_csv(f"Zone{selected_zone}_DWF.csv", parse_dates=["Time"])
        if "FlowDifference" in df_flow.columns:
            df_flow.rename(columns={"FlowDifference": "Measured Flow (L/s)"}, inplace=True)
        df_flow.set_index("Time", inplace=True)
    
        # Convert to Liters per 5-minute interval
        df_flow["Measured Flow (L)"] = df_flow["Measured Flow (L/s)"] * 60

        # Load Water Consumption
        df_cons = pd.read_csv(f"DWF_WaterConsumption_Zone{selected_zone}.csv", parse_dates=["Time"])
        df_cons.rename(columns={"Consumption": "Consumption (m³/h)"}, inplace=True)
        df_cons.set_index("Time", inplace=True)
        df_cons["Water Consumption (L)"] = df_cons["Consumption (m³/h)"] * 1000 / 12

        # Adjust Zone 8 Consumption Total
        if selected_zone == 8:
            df_cons["Water Consumption (L)"] *= 1500289.8 / df_cons["Water Consumption (L)"].sum()

        # Combine
        df_plot = pd.merge(
            df_flow[["Measured Flow (L)"]],
            df_cons[["Water Consumption (L)"]],
            left_index=True,
            right_index=True,
            how="inner"
        )

        st.line_chart(df_plot)

    except Exception as e:
        st.error(f"Failed to load data for Zone {selected_zone}: {e}")

    st.title("SWMM Simulated vs Measured Flow")
    st.markdown("""
    This method compares **SWMM-simulated dry weather flow** with **actual sensor-measured dry weather flow**.  
    Any excess in measured flow is interpreted as **infiltration (I/I)**.
    """)

    sensor_labels = {
        "1 Sensor 82952": 1,
        "2 Sensor 21877": 2,
        "3 Sensor 21886": 3,
        "4 Sensor 21856": 4,
        "6 Sensor 21897": 6,
        "7 Sensor 82503": 7,
        "8 Sensor 87090": 8,
    }

    volume_threshold = 10000
    percent_threshold = 15
    ii_results = []

    for label, sensor_num in sensor_labels.items():
        try:
            # Load expected + measured 
            swmm_file = f"SWMMdwf_SENSOR{sensor_num}.xlsx"
            measured_file = f"DWF_SENSOR{sensor_num}_fixed.csv"

            df_swmm = pd.read_excel(swmm_file)
            df_swmm["Time"] = pd.to_datetime(df_swmm["Date"].astype(str) + " " + df_swmm["Time"].astype(str), errors="coerce")
            df_swmm.set_index("Time", inplace=True)
            df_swmm.rename(columns={"(CMS)": "Expected (SWMM)"}, inplace=True)
            df_swmm = df_swmm[["Expected (SWMM)"]]

            df_measured = pd.read_csv(measured_file, parse_dates=["Time"])
            df_measured.set_index("Time", inplace=True)
            if "FlowDifference" in df_measured.columns:
                df_measured.rename(columns={"FlowDifference": "Measured (Sensor)"}, inplace=True)
            elif "Flow" in df_measured.columns:
                df_measured.rename(columns={"Flow": "Measured (Sensor)"}, inplace=True)


            df_measured = df_measured[["Measured (Sensor)"]]
            df_combined = pd.merge(df_swmm, df_measured, left_index=True, right_index=True, how="inner")
            df_combined["Infiltration (L/s)"] = (df_combined["Measured (Sensor)"] - df_combined["Expected (SWMM)"]).clip(lower=0)

            total_expected = df_combined["Expected (SWMM)"].sum() * 60
            total_measured = df_combined["Measured (Sensor)"].sum() * 60
            total_ii = df_combined["Infiltration (L/s)"].sum() * 60
            ii_percent = (total_ii / total_measured) * 100 if total_measured > 0 else 0

            status = "🚨 High I/I" if total_ii > volume_threshold or ii_percent > percent_threshold else "✅ Normal"

            ii_results.append({
                "Sensor": label,
                "Expected Flow (L)": round(total_expected, 2),
                "Measured Flow (L)": round(total_measured, 2),
                "I/I Volume (L)": round(total_ii, 2),
                "I/I %": round(ii_percent, 1),
                "Status": status
            })


        except Exception as e:
            st.warning(f"{label}: Error processing – {e}")


    df_result = pd.DataFrame(ii_results)

    styled_df = df_result.style.applymap(
        lambda val: 'color: red; font-weight: bold' if val == "🚨 High I/I" else '',
        subset=["Status"]
    )

    st.dataframe(styled_df)
    
    #st.bar_chart(df_result.set_index("Sensor")["I/I Volume (L)"])

    
    df_swmm_result = pd.DataFrame(ii_results)


    sensor_labels = {
        "1 Sensor 82952": 1,
        "2 Sensor 21877": 2,
        "3 Sensor 21886": 3,
        "4 Sensor 21856": 4,
        "6 Sensor 21897": 6,
        "7 Sensor 82503": 7,
        "8 Sensor 87090": 8,
        "9 Sensor 87108": 9
    }

    selected_label = st.selectbox("Choose a sensor:", list(sensor_labels.keys()))
    sensor_num = sensor_labels[selected_label]

    swmm_file = f"SWMMdwf_SENSOR{sensor_num}.xlsx"  #
    measured_file = f"DWF_SENSOR{sensor_num}_fixed.csv"

    try:
        # --- Load Expected (SWMM) ---
        df_swmm = pd.read_excel(swmm_file)
        df_swmm["Time"] = pd.to_datetime(
            df_swmm["Date"].astype(str) + " " + df_swmm["Time"].astype(str),
            errors="coerce"
        )
        df_swmm.set_index("Time", inplace=True)
        df_swmm.rename(columns={"(CMS)": "Expected (SWMM)"}, inplace=True)
        df_swmm = df_swmm[["Expected (SWMM)"]]

        # --- Load Measured ---
        df_measured = pd.read_csv(measured_file, parse_dates=["Time"])
        df_measured.set_index("Time", inplace=True)
        if "FlowDifference" in df_measured.columns:
            df_measured.rename(columns={"FlowDifference": "Measured (Sensor)"}, inplace=True)
        elif "Flow" in df_measured.columns:
            df_measured.rename(columns={"Flow": "Measured (Sensor)"}, inplace=True)
        df_measured = df_measured[["Measured (Sensor)"]]

        # --- Merge and Plot ---
        df_combined = pd.merge(df_swmm, df_measured, left_index=True, right_index=True, how="inner")
        st.line_chart(df_combined)

    except Exception as e:
        st.error(f"Error processing sensor {sensor_num}: {e}")

    st.write("     ")
    st.write("     ")
    st.write("     ")


    
    st.title("Minimum Night Flow (MNF) – Infiltration Estimation")
    st.markdown("Calculates average night-time flow (00:00–05:00) during dry weather to estimate base infiltration.")

    zones = [1, 2, 3, 4, 6, 7, 8]
    mnf_results = []
    zone_data = {}  # Store DataFrames and values for plotting

    for zone in zones:
        try:
            # Load DWF sensor data
            df = pd.read_csv(f"Zone{zone}_DWF.csv", parse_dates=["Time"])
            df.set_index("Time", inplace=True)

            # Use correct flow column
            if "FlowDifference" in df.columns:
                df.rename(columns={"FlowDifference": "Flow"}, inplace=True)
            elif "Flow" not in df.columns:
                st.warning(f"Zone {zone}: No flow column found.")
                continue

            # Filter for night hours
            df_night = df.between_time("00:00", "05:00")

            # Calculate MNF stats
            mnf_avg = df_night["Flow"].mean()
            mnf_min = df_night["Flow"].min()

            # Store for summary
            threshold = 10  # m³/day
            status = "🚨 High I/I" if mnf_avg * 3600 * 5 / 1000 > threshold else "✅ Normal"

            mnf_results.append({
                "Zone": f"Zone {zone}",
                "MNF (L/s)": round(mnf_avg, 3),
                "MNF (m³/day)": round(mnf_avg * 3600 * 5 / 1000, 2),
                "Status": status
            })

            # Store for plotting
            zone_data[zone] = {
                "df": df,
                "mnf_min": mnf_min,
                "mnf_avg": mnf_avg
            }

        except Exception as e:
            st.warning(f"Zone {zone}: Error – {e}")

    # Display summary table
    df_result = pd.DataFrame(mnf_results)
    st.dataframe(df_result)


    # Show bar chart
    #st.bar_chart(df_result.set_index("Zone")["MNF (m³/day)"])

    # Let user pick a zone to plot
    if zone_data:
        selected_zone = st.selectbox("Select a Zone to View Flow Plot", options=zone_data.keys(), format_func=lambda x: f"Zone {x}")

        df_plot = zone_data[selected_zone]["df"]
        mnf_min_plot = zone_data[selected_zone]["mnf_min"]

        fig, ax = plt.subplots(figsize=(12, 5))
        df_plot["Flow"].plot(ax=ax, alpha=0.4, label="Measured Flow")
        ax.axhline(y=mnf_min_plot, color="orange", linestyle="--", label=f"Min Night Flow = {mnf_min_plot:.2f} l/s")
        ax.set_title(f"Sensor {selected_zone} - Minimum Night Flow (MNF)")
        ax.set_ylabel("Flow [l/s]")
        ax.set_xlabel("Time")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        st.write("     ")
        st.write("     ")
        st.write("     ")

    st.title("Dry Weather Flow VS Wet Weather Flow")
    st.markdown("""
    This analysis quantifies **infiltration and inflow caused by rainfall**.  
    It compares flow measured during **wet weather (WWF)** to **dry weather (DWF)** in each zone.  
    The difference is interpreted as I/I from rainfall.
    """)

    zones = [1, 2, 3, 4, 6, 7, 8]
    ii_results = []

    # Thresholds
    volume_threshold = 10000  # in L
    percent_threshold = 15    # in %

    for zone in zones:
        try:
            # Load zone sensor files
            df_dwf = pd.read_csv(f"Zone{zone}_DWF.csv", parse_dates=["Time"])
            df_wwf = pd.read_csv(f"Zone{zone}_WWF.csv", parse_dates=["Time"])

            # Use correct flow column
            if "Flow" in df_dwf.columns:
                df_dwf.rename(columns={"Flow": "Flow (L/s)"}, inplace=True)
            elif "FlowDifference" in df_dwf.columns:
                df_dwf.rename(columns={"FlowDifference": "Flow (L/s)"}, inplace=True)

            if "Flow" in df_wwf.columns:
                df_wwf.rename(columns={"Flow": "Flow (L/s)"}, inplace=True)
            elif "FlowDifference" in df_wwf.columns:
                df_wwf.rename(columns={"FlowDifference": "Flow (L/s)"}, inplace=True)

            # Estimate total volume (sum of flow in L/s * 60s / number of days)
            dwf_volume = df_dwf["Flow (L/s)"].sum() * 60 / 4
            wwf_volume = df_wwf["Flow (L/s)"].sum() * 60 / 7

            ii_volume = wwf_volume - dwf_volume
            ii_percent = (ii_volume / wwf_volume) * 100 if wwf_volume > 0 else 0

            # Flag high I/I
            status = "🚨 High I/I" if ii_volume > volume_threshold or ii_percent > percent_threshold else "✅ Normal"

            ii_results.append({
                "Zone": f"Zone {zone}",
                "DWF Volume (L)": round(dwf_volume, 2),
                "WWF Volume (L)": round(wwf_volume, 2),
                "I/I Volume (L)": round(ii_volume, 2),
                "I/I %": round(ii_percent, 1),
                "Status": status
            })

        except Exception as e:
            st.warning(f"Zone {zone}: Error reading data – {e}")

    # Build and display results table
    df_result = pd.DataFrame(ii_results)

    # Optional: style status column
    styled_df = df_result.style.applymap(
        lambda val: 'color: red; font-weight: bold' if val == "🚨 High I/I" else '',
        subset=["Status"]
    )
    st.dataframe(styled_df)

    # Chart
    #st.markdown("### I/I Volume by Zone")
    #st.bar_chart(df_result.set_index("Zone")["I/I Volume (L)"])

    df_wwf_result = pd.DataFrame(ii_results)

        # Style and fonts
    plt.style.use("default")
    plt.rcParams.update({"font.size": 12})

    zones = [1, 2, 3, 4, 6, 7, 8]
    zone = st.selectbox("Select Zone", zones)

    try:
        # --- Load data ---
        dwf = pd.read_csv(f"Zone{zone}_DWF.csv")
        bsf_raw = pd.read_csv(f"DWF_WaterConsumption_Zone{zone}.csv")
        wwf = pd.read_csv(f"Zone{zone}_WWF.csv")
        precip = pd.read_excel("Steinrusten(WWF).xlsx")

        # Time conversion
        dwf["Time"] = pd.to_datetime(dwf["Time"])
        bsf_raw["Time"] = pd.to_datetime(bsf_raw["Time"])
        wwf["Time"] = pd.to_datetime(wwf["Time"])
        precip["Time"] = pd.to_datetime(precip["Time"])

        # GWI (min dry weather flow)
        gwi = dwf["FlowDifference"].min()

        # BSF: hourly pattern
        bsf_raw["Hour"] = bsf_raw["Time"].dt.hour
        bsf_pattern = bsf_raw.groupby("Hour")["Consumption"].mean()
        bsf_extended = np.tile(bsf_pattern.values, int(np.ceil(len(wwf)/24)))[:len(wwf)]
        bsf = pd.Series(bsf_extended, index=wwf.index)

        # RDII
        rdii = wwf["FlowDifference"] - gwi - bsf
        rdii = rdii.clip(lower=0)

        # Total flow contributions (averages)
        avg_gwi = gwi
        avg_bsf = bsf.mean()
        avg_rdii = rdii.mean()

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(10, 6))

        # GWI band
        ax.fill_between(wwf["Time"], 0, gwi, color="#eeeeee", hatch=".", edgecolor="gray", linewidth=0.5)

        # BSF
        ax.fill_between(wwf["Time"], gwi, gwi + bsf, color="#009de0", label="BSF")

        # RDII
        ax.fill_between(wwf["Time"], gwi + bsf, gwi + bsf + rdii, color="#b0dffb", label="RDII")

        # Precipitation (secondary axis)
        ax2 = ax.twinx()
        ax2.bar(precip["Time"], precip["Level"], width=0.03, color="lightblue", align="center", zorder=0)
        ax2.set_ylim(ax2.get_ylim()[::-1])  # Invert Y axis
        ax2.set_yticks([])
        ax2.set_ylabel("")

        # Annotations
        ax.text(wwf["Time"].iloc[len(wwf)//3], gwi / 2, "GWI", ha="center", va="center", weight="bold")
        ax.text(wwf["Time"].iloc[len(wwf)//3], gwi + bsf.mean()/2, "BSF", ha="center", va="center", color="white", weight="bold")
        ax.text(wwf["Time"].iloc[len(wwf)//3], gwi + bsf.mean() + rdii.mean()/2, "RDII", ha="center", va="center", weight="bold")

        ax.set_title("Erdal I/I", fontweight="bold")
        ax.set_xlabel("Time")
        ax.set_ylabel("Flow [l/s]")

        # Hide right spine and grid
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.grid(False)

        # --- Right-side Value Labels ---
        label_x = wwf["Time"].iloc[-1]
        ax.text(label_x, gwi / 2, f"{avg_gwi:.1f} l/s", va="center", ha="left", fontsize=11)
        ax.text(label_x, gwi + avg_bsf/2, f"{avg_bsf:.1f} l/s", va="center", ha="left", color="white", fontsize=11)
        ax.text(label_x, gwi + avg_bsf + avg_rdii/2, f"{avg_rdii:.1f} l/s", va="center", ha="left", fontsize=11)

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error generating Erdal I/I plot for Zone {zone}: {e}")


    
if page == "Zone Map and Recommendation":
    st.title("Zone Map")
    st.write("This pages illustrates with a colour coded system showing the I/I rates in the different zones. The red zones indicates a high level of I/I, needing further investigation.")

    # Status data
    zone_data = [
        {"Zone": "ZONE1", "Status": "Low"},
        {"Zone": "ZONE2", "Status": "Low"},
        {"Zone": "ZONE3", "Status": "High"},
        {"Zone": "ZONE4", "Status": "Low"},
        {"Zone": "ZONE6", "Status": "Medium"},
        {"Zone": "ZONE7", "Status": "High"},
        {"Zone": "ZONE8", "Status": "High"},
        {"Zone": "ZONE9", "Status": "High"},
        # Add more zones here
    ]

    # Color mapping
    def status_to_color(status):
        return {
            "High": (255, 0, 0, 128),      # Red w
            "Medium": (255, 165, 0, 128),  # Orange
            "Low": (0, 128, 0, 128),       # Green
        }.get(status, (128, 128, 128, 128))  # Default gray

    # Approximate polygon coordinates (in pixel space on the image)
    zone_polygons = {
        "ZONE1": [(10, 156), (103, 237), (122, 322), (84, 455), (8, 424)],
        "ZONE2": [(2, 5), (275, 3), (351, 134), (318, 165), (95, 158), (9, 88)],
        "ZONE3": [(243, 171), (325, 182), (363, 145), (467, 180), (510, 245), (532, 376)],
        "ZONE4": [(152, 239), (245, 234), (402, 346), (351, 370), (310, 505), (250, 518), (172, 470), (132, 396), (161, 324)],
        "ZONE6": [(356, 396), (431, 359), (586, 485), (580, 520), (481, 538), (541, 632), (474, 645), (475, 610), (388, 612), (343, 597), (317, 535)],
        "ZONE7": [(338, 47), (551, 27), (590, 259), (671, 326), (657, 422), (861, 472), (826, 616), (730, 684), (570, 629), (512, 538), (598, 529), (605, 444), (562, 398), (503, 197), (420, 147)],
        "ZONE8": [(790, 690), (812, 680), (826, 719), (822, 784), (784, 721)],
        "ZONE9": [(491, 666), (611, 656), (777, 719), (832, 802), (633, 778), (497, 710)],
    }

    # Load the image
    image_path = "ERDAL.png"
    image = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(image, 'RGBA')

    # Draw polygons with fill based on status
    for zone in zone_data:
        name = zone["Zone"]
        status = zone["Status"]
        coords = zone_polygons.get(name)
    
        if coords:
            fill_color = status_to_color(status) 
            draw.polygon(coords, fill=status_to_color(status), outline="black")
        
            # Draw label at approximate center
            center_x = sum(p[0] for p in coords) // len(coords)
            center_y = sum(p[1] for p in coords) // len(coords)
            draw.text((center_x - 20, center_y - 10), name, fill="black")

    # Show the result
    st.image(image, caption="I/I Zone Map", use_column_width=True)


    st.subheader("Recommendation")
    st.write("The analysis revealed that **zones 3, 6, 7, and 8** are particularly affected by high I/I, as evidenced by both elevated MNF values and RDII percentages. For instance, Zone 3 exhibited an RDII percentage of 99.4%, indicating that nearly all wet weather flow in this area is attributable to I/I rather than sanitary flow. Such high values not only point to insufficient separation between stormwater and sanitary systems but also signal the urgent need for targeted interventions, such as CCTV inspections and smoke testing, to locate and remediate sources of I/I.")


elif page == "Steinrusten Rain Gauge":
    st.title("Steinrusten rain Gauge")
    st.write("Under, we can see the rain and water consumption from the two analysis periods")
    st.write("     ")
    st.subheader("Steinrusten Rainfall")

        # Let the user choose rain type (default = WWF)
    rain_type = st.radio("Select rain period:", ["WWF", "DWF"], index=0)  # WWF is default

    # Load the selected file
    if rain_type == "DWF":
        rain_file = "RainDataDWF.csv"
    else:
        rain_file = "RainDataWWF.csv"

    try:
        # Load the selected rain data
        rain_df = pd.read_csv(rain_file, parse_dates=["Time"])
        rain_df.set_index("Time", inplace=True)

        # Plot
        st.write(f"{rain_type} Rainfall")
        st.line_chart(rain_df["Level"])  

    except FileNotFoundError:
        st.error(f"{rain_file} not found.")
    except Exception as e:
        st.error(f"Error loading rain data: {e}")

elif page == "Erdal Water Consumption":
    st.title("Erdal Water Consumption")
    st.write("Under, we can see the rain and water consumption from the analysed periods.")
    st.write("     ")
    st.write("     ")  
    # Choose period, default to DWF
    flow_type = st.radio("Select period:", ["DWF", "WWF"], index=0)

    # Zone selection (skip Zone 5)
    zones = [1, 2, 3, 4, 6, 7, 8, 9]
    selected_zone = st.selectbox("Select zone:", zones)

    # File names
    total_file = f"WaterConsumption{flow_type}.csv"
    zone_file = f"{flow_type}_WaterConsumption_Zone{selected_zone}.csv"

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Total Water Consumption ({flow_type})")
        try:
            df_total = pd.read_csv(total_file, parse_dates=["Date"])
            df_total.rename(columns={"Date": "Time", "m3t": "Water Use (m³)"}, inplace=True)
            df_total.set_index("Time", inplace=True)
            st.line_chart(df_total["Water Use (m³)"])
        except Exception as e:
            st.error(f"Error loading {total_file}: {e}")

    with col2:
        st.subheader(f"Zone {selected_zone} – Water Consumption ({flow_type})")
        try:
            df_zone = pd.read_csv(zone_file, parse_dates=["Time"])
            df_zone.rename(columns={"Consumption": "Water Use (m³)"}, inplace=True)
            df_zone.set_index("Time", inplace=True)
            st.line_chart(df_zone["Water Use (m³)"])
        except Exception as e:
            st.error(f"Error loading {zone_file}: {e}")

elif page == "Sensor and Zone Data":
    st.title("Sensor and Zone Data")
    st.write("Under, we can see the fixed sensordata, already gone through data validation")

    st.subheader("Sensor Data ")
   
   # Map readable sensor labels to numbers
    sensor_labels = {
        "Sensor1: 82952": 1,
        "Sensor2: 21877": 2,
        "Sensor3: 21886": 3,
        "Sensor4: 21856": 4,
        "Sensor5: 21887": 5,
        "Sensor6: 21897": 6,
        "Sensor7: 82503": 7,
        "Sensor8: 87090": 8,
        "Sensor9: 87108": 9,
    }

    selected_label = st.selectbox("Choose a sensor:", list(sensor_labels.keys()))
    sensor_num = sensor_labels[selected_label]

    # Build file paths
    dwf_file = f"DWF_SENSOR{sensor_num}_fixed.csv"
    wwf_file = f"WWF_SENSOR{sensor_num}_fixed.csv"
    rainfall_file = "RainDataWWF.csv"

    try:
        # Load CSVs
        df_dwf = pd.read_csv(dwf_file, parse_dates=["Time"])
        df_wwf = pd.read_csv(wwf_file, parse_dates=["Time"])

        df_dwf.set_index("Time", inplace=True)
        df_wwf.set_index("Time", inplace=True)

        # Allow user to pick variables
        shared_vars = [col for col in df_dwf.columns if col in df_wwf.columns]
        selected_vars = st.multiselect("Select variables to plot:", shared_vars, default=["Flow"])

        if selected_vars:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("DWF")
                fig1, ax1 = plt.subplots()
                df_dwf[selected_vars].plot(ax=ax1)
                ax1.set_title(f"DWF – Sensor {sensor_num}")
                ax1.set_ylabel("Value")
                st.pyplot(fig1)

                # Load rainfall data
            try:
                rainfall_df = pd.read_csv(rainfall_file, parse_dates=["Time"])
                rainfall_df.set_index("Time", inplace=True)
                rainfall_df = rainfall_df.resample("H").mean()

                # Align with WWF time range only if WWF data exists
                if not df_wwf.empty:
                    rainfall_df = rainfall_df.loc[df_wwf.index.min():df_wwf.index.max()]
                else:
                    st.warning("WWF data is empty; rainfall will not be shown.")
                    rainfall_df = None

            except FileNotFoundError:
                st.warning(f"Rainfall data file '{rainfall_file}' not found.")
                rainfall_df = None
            except Exception as e:
                st.warning(f"Could not load rainfall data: {e}")
                rainfall_df = None
                
            with col2:
                st.subheader("WWF")
                fig2, ax2 = plt.subplots()
                df_wwf[selected_vars].plot(ax=ax2)
                ax2.set_title(f"WWF – Sensor {sensor_num}")
                ax2.set_ylabel("Flow")
                st.pyplot(fig2)

                # Add rainfall on second y-axis if data exists
            if rainfall_df is not None and not rainfall_df.empty:
                ax3 = ax2.twinx()
                rainfall_df["Level"].plot(ax=ax3, color="blue", linestyle="--", label="Rainfall Level")
                ax3.set_ylabel("Rainfall Level", color="blue")
                ax3.tick_params(axis='y', labelcolor="blue")

                fig2.autofmt_xdate()
                st.pyplot(fig2)

           

    except FileNotFoundError:
        st.error(f"One or both files not found for Sensor {sensor_num}.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

    st.write("   ")
    st.write("   ")
    st.write("   ")
    st.write("   ")
    st.write("   ")
    st.write("   ")
    st.write("   ")
    st.write("   ")
    st.write("   ")
    st.write("   ")
    st.write("   ")
    st.write("   ")
    st.write("   ")
    st.write("   ")
    st.write("   ")
    st.subheader("Netto Zone Flow")

    # Available zones 
    zones = [1, 2, 3, 4, 6, 7, 8, 9]
    selected_zone = st.selectbox("Select a Zone:", zones)

    # Build filenames
    dwf_file = f"Zone{selected_zone}_DWF.csv"
    wwf_file = f"Zone{selected_zone}_WWF.csv"
    rainfall_file = "RainDataWWF.csv"

    try:
        # Load DWF and WWF data
        df_dwf = pd.read_csv(dwf_file, parse_dates=["Time"])
        df_wwf = pd.read_csv(wwf_file, parse_dates=["Time"])

        df_dwf.rename(columns={"FlowDifference": "Flow"}, inplace=True)
        df_wwf.rename(columns={"FlowDifference": "Flow"}, inplace=True)

        df_dwf.set_index("Time", inplace=True)
        df_wwf.set_index("Time", inplace=True)

        # Load rainfall data
        try:
            rainfall_df = pd.read_csv(rainfall_file, parse_dates=["Time"])
            rainfall_df.set_index("Time", inplace=True)
            rainfall_df = rainfall_df.resample("H").mean()

            # Align with WWF time range only if WWF data exists
            if not df_wwf.empty:
                rainfall_df = rainfall_df.loc[df_wwf.index.min():df_wwf.index.max()]
            else:
                st.warning("WWF data is empty; rainfall will not be shown.")
                rainfall_df = None

        except FileNotFoundError:
            st.warning(f"Rainfall data file '{rainfall_file}' not found.")
            rainfall_df = None
        except Exception as e:
            st.warning(f"Could not load rainfall data: {e}")
            rainfall_df = None

        st.write(f"Zone {selected_zone} – DWF and WWF")

        
        st.subheader("DWF")
        fig1, ax1 = plt.subplots()
        df_dwf["Flow"].plot(ax=ax1, color="green")
        ax1.set_title(f"Zone {selected_zone} – DWF")
        ax1.set_ylabel("Flow")
        ax1.set_xlabel("Time")
        st.pyplot(fig1)

        st.subheader("WWF")
        fig2, ax2 = plt.subplots()

        df_wwf["Flow"].plot(ax=ax2, color="green", label="WWF Flow")
        ax2.set_title(f"Zone {selected_zone} – WWF and Rainfall")
        ax2.set_ylabel("WWF Flow", color="green")
        ax2.set_xlabel("Time")
        ax2.tick_params(axis='y', labelcolor="green")

        # Add rainfall on second y-axis if data exists
        if rainfall_df is not None and not rainfall_df.empty:
            ax3 = ax2.twinx()
            rainfall_df["Level"].plot(ax=ax3, color="blue", linestyle="--", label="Rainfall Level")
            ax3.set_ylabel("Rainfall Level", color="blue")
            ax3.tick_params(axis='y', labelcolor="blue")

        fig2.autofmt_xdate()
        st.pyplot(fig2)

    except FileNotFoundError:
        st.error("One or both CSV files not found.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
