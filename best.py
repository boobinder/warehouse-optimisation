import pandas as pd
import numpy as np
import folium
from geopy.distance import geodesic
from sklearn.cluster import KMeans
import openrouteservice
import matplotlib.pyplot as plt
import logging

# Load datasets
warehouse_df = pd.read_csv("Warehouse_with_Coordinates.csv")
january_df = pd.read_csv("Stockists_with_Coordinates.csv")
TOT_df = pd.read_csv("Sheet 3-TOT-1.csv")  # Cost per km data

# OpenRouteService API client
client = openrouteservice.Client(key='5b3ce3597851110001cf6248256c9a2abceb4c808de1a8386485d81d')  

# Extract stockist coordinates
stockist_coords = january_df[['Latitude', 'Longitude']].values

# Apply K-Means clustering for two warehouses
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(stockist_coords)
january_df['Cluster'] = kmeans.labels_

# Compute optimized warehouse locations (centroids)
optimized_warehouses = pd.DataFrame(kmeans.cluster_centers_, columns=['Latitude', 'Longitude'])


from geopy.geocoders import Nominatim
from geopy.distance import geodesic

# Initialize geocoder (use a user_agent)
geolocator = Nominatim(user_agent="warehouse_optimizer")

def get_location_name(lat, lon, max_retries=3):
    """Fetch location name or nearest named town."""
    location = None
    try:
        location = geolocator.reverse((lat, lon), exactly_one=True, language="en")
        if location and location.address:
            return location.address.split(",")[0]  # Return primary name (e.g., town)
    except Exception as e:
        logging.warning(f"Reverse geocoding failed: {e}")

    # Fallback: Find nearest named stockist town within 30 km
    min_distance = float("inf")
    nearest_town = "Unnamed Location"
    for _, stockist in january_df.iterrows():
        dist = geodesic((lat, lon), (stockist["Latitude"], stockist["Longitude"])).km
        if dist < min_distance and dist <= 30:  # Search radius = 30 km
            min_distance = dist
            nearest_town = stockist["Town"]
    return f"{nearest_town} (Nearest Named Location)"

# Add location names to optimized warehouses
optimized_warehouses["Location_Name"] = optimized_warehouses.apply(
    lambda row: get_location_name(row["Latitude"], row["Longitude"]), axis=1
)

# Save updated CSV
optimized_warehouses.to_csv("addy.csv", index=False)

# Configure logging for improved error handling
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to calculate delivery cost using OpenRouteService
def calculate_cost(stockist_df, warehouse_coords):
    total_cost = 0
    for _, row in stockist_df.iterrows():
        stockist_location = (row['Latitude'], row['Longitude'])
        distances = []
        
        for wh in warehouse_coords:
            try:
                route = client.directions(coordinates=[(wh[1], wh[0]), (stockist_location[1], stockist_location[0])], 
                                          profile='driving-car', format='geojson')
                distance_km = route['routes'][0]['segments'][0]['distance'] / 1000  # Convert meters to km
                distances.append(distance_km)
            except:
                distances.append(geodesic(stockist_location, tuple(wh)).km)  # Fallback to geodesic distance
        
        min_distance = min(distances)
        cost_per_km = TOT_df['Charge(in Rs.)'].mean()
        total_cost += min_distance * cost_per_km
    return total_cost

# Cost before optimization (using original warehouses)
original_warehouse_coords = warehouse_df[['Latitude', 'Longitude']].values
original_cost = calculate_cost(january_df, original_warehouse_coords)

# Cost after optimization (using optimized warehouses)
optimized_warehouse_coords = optimized_warehouses[['Latitude', 'Longitude']].values
optimized_cost = calculate_cost(january_df, optimized_warehouse_coords)


# Generate interactive map
map_ = folium.Map(location=[np.mean(stockist_coords[:, 0]), np.mean(stockist_coords[:, 1])], zoom_start=6)

# Initialize Dark Mode Map
map_ = folium.Map(
    location=[np.mean(stockist_coords[:, 0]), np.mean(stockist_coords[:, 1])],
    zoom_start=6,
    tiles="CartoDB Positron",
    attr="© OpenStreetMap contributors | CartoDB Positron"
)

# Add stockist markers (blue)
for _, row in january_df.iterrows():
    folium.CircleMarker(
    location=[row['Latitude'], row['Longitude']],
    radius=6,  # Increase size
    color='cyan',
    fill=True,
    fill_color='cyan',
    fill_opacity=0.9  # Adjust opacity for better visibility
).add_to(map_)
    
# Modify the original warehouse markers section
for idx, row in warehouse_df.iterrows():
    # Customize based on warehouse number
    if idx == 0:  # Warehouse 1
        icon_color = 'red'
        icon_type = 'home'
        prefix = 'fa'  # Font Awesome icons
        wh_label = 'Original W1'
    else:  # Warehouse 2
        icon_color = 'blue'
        icon_type = 'industry'
        prefix = 'fa'
        wh_label = 'Original W2'
    
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        icon=folium.Icon(
            color=icon_color,
            icon=icon_type,
            prefix=prefix,
            icon_color='white'  # White icon on colored background
        ),
        popup=f"""
            <b>{wh_label}</b><br>
            <i>{row['Town']}</i><br>
            Coordinates: {row['Latitude']:.4f}, {row['Longitude']:.4f}
        """,
        tooltip=f"{wh_label} - {row['Town']}",
        marker_kwargs={'opacity': 0.9}
    ).add_to(map_)


# Add optimized warehouse markers (green) - MODIFIED SECTION
for _, row in optimized_warehouses.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        icon=folium.Icon(color='green', icon='ok-sign'),
        popup=f"Optimized Warehouse: {row['Location_Name']}",  # Added location name
        tooltip=f"Click for details: {row['Location_Name']}"   # Added hover tooltip
    ).add_to(map_)


# Extract original warehouse coordinates (W1 and W2)
w1_coords = warehouse_df.iloc[0][['Latitude', 'Longitude']].tolist()
w2_coords = warehouse_df.iloc[1][['Latitude', 'Longitude']].tolist()

# Assign each stockist to their original warehouse (W1/W2)
january_df['Original_Warehouse'] = january_df.apply(
    lambda row: 'W1' if geodesic((row['Latitude'], row['Longitude']), w1_coords).km < 
                       geodesic((row['Latitude'], row['Longitude']), w2_coords).km 
                else 'W2', 
    axis=1
)


# Function to add styled routes
def add_routes_to_map(map_, stockist_df, warehouse_coords, color, weight, dash_array, label):
    """Add routes with custom styles to the map."""
    for _, row in stockist_df.iterrows():
        stockist_loc = (row['Latitude'], row['Longitude'])
        try:
            # Get route from OpenRouteService
            route = client.directions(
                coordinates=[(warehouse_coords[1], warehouse_coords[0]),  # WH coords
                          (stockist_loc[1], stockist_loc[0])],  # Stockist coords
                profile='driving-car',
                format='geojson'
            )
            route_points = [(point[1], point[0]) for point in 
                           route['routes'][0]['geometry']['coordinates']]
            distance = route['routes'][0]['segments'][0]['distance']/1000
        except Exception as e:
            # Fallback to straight-line distance
            route_points = [warehouse_coords, stockist_loc]
            distance = geodesic(warehouse_coords, stockist_loc).km

        # Add polyline to map
        folium.PolyLine(
            locations=route_points,
            color=color,
            weight=weight,
            dash_array=dash_array,
            opacity=0.7,
            popup=f"{label} Route: {distance:.1f} km",
            tooltip=row['Town']
        ).add_to(map_)

# Add Original W1 Routes (Red Dashed)
w1_stockists = january_df[january_df['Original_Warehouse'] == 'W1']
add_routes_to_map(map_, w1_stockists, w1_coords, 'red', 2, '5,5', 'Original W1')

# Add Original W2 Routes (Blue Dashed)
w2_stockists = january_df[january_df['Original_Warehouse'] == 'W2']
add_routes_to_map(map_, w2_stockists, w2_coords, 'blue', 2, '5,5', 'Original W2')

# Add Optimized Routes (Green Solid)
for _, opt_wh in optimized_warehouses.iterrows():
    opt_coords = (opt_wh['Latitude'], opt_wh['Longitude'])
    cluster_stockists = january_df[january_df['Cluster'] == _]
    add_routes_to_map(map_, cluster_stockists, opt_coords, 'green', 3, None, 'Optimized')

# Update the legend HTML to match
legend_html = """
<div style="position: fixed; bottom: 20px; left: 20px; z-index: 1000; 
            background: white; padding: 10px; border-radius: 5px; box-shadow: 0 0 5px rgba(0,0,0,0.2)">
    <h4>Legend</h4>
    <p style="margin:2px"><i class="fa fa-home" style="color: red"></i> Original W1</p>
    <p style="margin:2px"><i class="fa fa-industry" style="color: blue"></i> Original W2</p>
    <p style="margin:2px"><i class="fa fa-ok-sign" style="color: green"></i> Optimized WH</p>
    <p style="margin:2px; color: cyan">● Stockists</p>
    <p style="margin:2px; color: red">--- W1 Routes</p>
    <p style="margin:2px; color: blue">--- W2 Routes</p>
    <p style="margin:2px; color: green">➔ Optimized Routes</p>
</div>
"""
map_.get_root().html.add_child(folium.Element(legend_html))

# Save the map with updated markers and styles

map_.save("optimized_warehouse_map.html")
print("Updated map with custom marker colors and route styles saved as 'optimized_warehouse_map.html'.")



import matplotlib.pyplot as plt
import numpy as np
from geopy.distance import geodesic

# Stockist names (X-axis)
stockist_names = january_df['Town'].values

# Distances and Costs
original_distances = []
optimized_distances = []

for _, row in january_df.iterrows():
    stockist_location = (row['Latitude'], row['Longitude'])
    
    # Distance to nearest original warehouse
    original_dists = [geodesic(stockist_location, (wh[0], wh[1])).km for wh in original_warehouse_coords]
    original_distances.append(min(original_dists))
    
    # Distance to nearest optimized warehouse
    optimized_dists = [geodesic(stockist_location, (wh[0], wh[1])).km for wh in optimized_warehouse_coords]
    optimized_distances.append(min(optimized_dists))

# Convert distances to costs
cost_per_km = TOT_df['Charge(in Rs.)'].mean()
original_costs = np.array(original_distances) * cost_per_km
optimized_costs = np.array(optimized_distances) * cost_per_km

# Simulated frequency data (Replace with actual frequency data if available)
original_frequencies = np.random.randint(5, 30, len(stockist_names))  # Simulated
optimized_frequencies = np.random.randint(5, 30, len(stockist_names))  # Simulated

# Create the figure and axis
fig, ax1 = plt.subplots(figsize=(14, 7))  # Define ax1
ax2 = ax1.twinx()  # Define ax2 for dual-axis plotting

import os

# Create "assets" folder if it doesn't exist
if not os.path.exists("assets"):
    os.makedirs("assets")

# Bar chart for costs
width = 0.4  # Width of bars
x = np.arange(len(stockist_names))

bars1 = ax1.bar(x - width/2, original_costs, width, label="Before Optimization (₹)", color='red', alpha=0.7)
bars2 = ax1.bar(x + width/2, optimized_costs, width, label="After Optimization (₹)", color='green', alpha=0.7)

# Format bar chart axis
ax1.set_xlabel("Stockists", fontsize=12)
ax1.set_ylabel("Delivery Cost (₹)", fontsize=12, color='black')
ax1.set_xticks(x)
ax1.set_xticklabels(stockist_names, rotation=45, ha='right', fontsize=10)  # Adjusted for better readability
ax1.grid(axis='y', linestyle="--", alpha=0.5)

# Add cost labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 200, f"₹{yval:.0f}", 
                 ha='center', fontsize=9, color='black')
        
        
        # Save the bar chart as a PNG file

# Call legend from ax1 (not plt)
ax1.legend(
    loc="upper left",
    bbox_to_anchor=(1.05, 1),  # Moves legend outside the plot
    fontsize=12,
    frameon=True,
    edgecolor='black',
    facecolor='white',
    title="Delivery Cost Comparison",
    title_fontsize=13
)


plt.subplots_adjust(right=0.8)
plt.tight_layout()  # Ensures everything fits well
plt.title("Comparison of Delivery Costs Before and After Optimization")
plt.savefig("assets/Delivery_Cost_Comparison.png", dpi=300, bbox_inches='tight')
plt.show()
       


# Clean column names by stripping extra spaces
january_df.columns = january_df.columns.str.strip()

# Drop rows with missing frequency values
january_df.dropna(subset=['Frequency(W1)', 'Frequency(W2)'], inplace=True)

# Extract data after cleaning
stockist_names = january_df['Town'].values
w1_frequencies = january_df['Frequency(W1)'].values
w2_frequencies = january_df['Frequency(W2)'].values

import matplotlib.pyplot as plt

# Create the figure and axis
plt.figure(figsize=(14, 7))

# Line plots for delivery frequency
plt.plot(stockist_names, w1_frequencies, color='blue', marker='o', linestyle='-', linewidth=2, markersize=6, label="W1 Frequency")
plt.plot(stockist_names, w2_frequencies, color='orange', marker='s', linestyle='--', linewidth=2, markersize=6, label="W2 Frequency")

# Add frequency values above points
for i in range(len(stockist_names)):
    plt.text(i, w1_frequencies[i] + 1, f"{w1_frequencies[i]}", ha='center', fontsize=9, color='blue')
    plt.text(i, w2_frequencies[i] + 1, f"{w2_frequencies[i]}", ha='center', fontsize=9, color='orange')

# Graph Formatting
plt.xlabel("Stockists", fontsize=12)
plt.ylabel("Delivery Frequency", fontsize=12)
plt.title("Stockist Delivery Frequency from Original Warehouses", fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.grid(axis='y', linestyle="--", alpha=0.5)
plt.legend(loc="upper right", fontsize=10)

plt.tight_layout()  # Prevent overlap
plt.savefig('assets/Stockist_Delivery_Frequency.png', dpi=300, bbox_inches='tight')
plt.show()

# Print final results
print(f"Original Cost: ₹{original_cost:.2f}")
print(f"Optimized Cost: ₹{optimized_cost:.2f}")
print("Optimization complete! Open 'optimized_warehouse_map.html' to view the map.")

import pandas as pd
import numpy as np
from geopy.distance import geodesic
import os
from datetime import datetime

# Function to calculate nearest warehouse
def find_nearest_warehouse(stockist_lat, stockist_lon, warehouses):
    distances = [geodesic((stockist_lat, stockist_lon), (wh[0], wh[1])).km for wh in warehouses]
    return warehouses[np.argmin(distances)]

# Enhanced Cost Calculation with Vectorized Operations
def calculate_cost(stockist_df, warehouse_coords, cost_per_km):
    distances = [
        min([geodesic((row['Latitude'], row['Longitude']), (wh[0], wh[1])).km for wh in warehouse_coords])
        for _, row in stockist_df.iterrows()
    ]
    return np.array(distances) * cost_per_km

# Ensure CSV files exist before reading
required_files = ["Warehouse_with_Coordinates.csv", "Stockists_with_Coordinates.csv", "Sheet 3-TOT-1.csv"]
for file in required_files:
    if not os.path.exists(file):
        print(f"Error: {file} not found. Please check the file path.")
        exit()

# Load datasets
warehouse_df = pd.read_csv("Warehouse_with_Coordinates.csv")
january_df = pd.read_csv("Stockists_with_Coordinates.csv")
TOT_df = pd.read_csv("Sheet 3-TOT-1.csv")
optimized_warehouse_coords = pd.read_csv("addy.csv")

# Cost Calculation
cost_per_km = TOT_df['Charge(in Rs.)'].mean()

# Original and Optimized Costs
original_costs = calculate_cost(january_df, warehouse_df[['Latitude', 'Longitude']].values, cost_per_km)
optimized_costs = calculate_cost(january_df, pd.read_csv("addy.csv")[['Latitude', 'Longitude']].values, cost_per_km)


# Create final results dataframe with correctly matched warehouses
final_results_df = pd.DataFrame({
    'Stockist': january_df['Town'],
    'Original Delivery Cost (₹)': original_costs.round(2),
    'Optimized Delivery Cost (₹)': optimized_costs.round(2),
    'Cost Savings (₹)': (original_costs - optimized_costs).round(2),
    'Reduction (%)': (((original_costs - optimized_costs) / original_costs) * 100).round(2),
    'Status': np.where((original_costs - optimized_costs) > 0, 'Improved', 'Loss')
})

# Add Total Row
total_row = pd.DataFrame({
    'Stockist': ['Total'],
    'Original Delivery Cost (₹)': [final_results_df['Original Delivery Cost (₹)'].sum().round(2)],
    'Optimized Delivery Cost (₹)': [final_results_df['Optimized Delivery Cost (₹)'].sum().round(2)],
    'Cost Savings (₹)': [final_results_df['Cost Savings (₹)'].sum().round(2)],
    'Reduction (%)': [''],
    'Status': ['']
})

# Concatenate and ensure numeric types
final_results_df = pd.concat([final_results_df, total_row], ignore_index=True)

# Convert numeric columns and handle errors
numeric_cols = ['Original Delivery Cost (₹)', 'Optimized Delivery Cost (₹)', 'Cost Savings (₹)']
for col in numeric_cols:
    final_results_df[col] = pd.to_numeric(final_results_df[col], errors='coerce')

# Format numeric columns
for col in numeric_cols:
    final_results_df[col] = final_results_df[col].apply(
        lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A"
    )

# Clean up Total row formatting
final_results_df.loc[final_results_df['Stockist'] == 'Total', numeric_cols] = final_results_df.loc[
    final_results_df['Stockist'] == 'Total', numeric_cols
].replace('N/A', '')

# Save the final CSV
final_results_df.to_csv("Improved_Warehouse_Optimization_Results.csv", index=False)