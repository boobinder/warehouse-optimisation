import pandas as pd
import numpy as np
import folium
from geopy.distance import geodesic
from sklearn.cluster import KMeans
import openrouteservice
import matplotlib.pyplot as plt
import logging
from scipy.spatial.distance import cdist
from geopy.geocoders import Nominatim
import os

# Configure logging for improved error handling
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure required files exist before reading
required_files = ["Warehouse_with_Coordinates.csv", "Stockists_with_Coordinates.csv", "Sheet 3-TOT-1.csv", "addy.csv"]
for file in required_files:
    if not os.path.exists(file):
        print(f"Error: {file} not found. Please check the file path.")
        exit()

# Load datasets
warehouse_df = pd.read_csv("Warehouse_with_Coordinates.csv")
january_df = pd.read_csv("Stockists_with_Coordinates.csv")
TOT_df = pd.read_csv("Sheet 3-TOT-1.csv")  # Cost per km data
best_warehouses_df = pd.read_csv("addy.csv")  # The first best warehouse locations

# OpenRouteService API client - Consider using environment variables for API keys
client = openrouteservice.Client(key='5b3ce3597851110001cf6248256c9a2abceb4c808de1a8386485d81d')  

# Extract stockist coordinates
stockist_coords = january_df[['Latitude', 'Longitude']].values

# Function to calculate delivery cost using OpenRouteService
def calculate_cost(stockist_df, warehouse_coords):
    total_cost = 0
    for _, row in stockist_df.iterrows():
        stockist_location = (row['Latitude'], row['Longitude'])
        distances = []
        
        for wh in warehouse_coords:
            try:
                # Keep consistent coordinate order for API
                warehouse_loc = (wh[1], wh[0])
                route = client.directions(
                    coordinates=[warehouse_loc, (stockist_location[1], stockist_location[0])], 
                    profile='driving-car', 
                    format='geojson'
                )
                distance_km = route['routes'][0]['segments'][0]['distance'] / 1000  # Convert meters to km
                distances.append(distance_km)
            except Exception as e:
                logging.warning(f"Route calculation failed: {e}. Falling back to geodesic.")
                try:
                    distances.append(geodesic(stockist_location, (wh[0], wh[1])).km)
                except Exception as fallback_error:
                    logging.error(f"Geodesic fallback failed: {fallback_error}")
                    # If both methods fail, use a large distance penalty
                    distances.append(1000)  # Large penalty distance
        
        min_distance = min(distances)
        cost_per_km = TOT_df['Charge(in Rs.)'].mean()
        total_cost += min_distance * cost_per_km
    return total_cost

# Implement a constrained K-means for second-best warehouses
def constrained_kmeans(data, k, avoid_coords, min_distance_km=50):
    """
    Run K-means but ensure centroids are at least min_distance_km away from avoid_coords
    
    Parameters:
    data - array of coordinates
    k - number of clusters
    avoid_coords - coordinates to avoid (best warehouse locations)
    min_distance_km - minimum distance from best warehouses in km
    """
    best_cost = float('inf')
    best_centroids = None
    best_labels = None
    
    # Try multiple random initializations
    for _ in range(10):
        # Initialize with random points
        random_indices = np.random.choice(len(data), k, replace=False)
        initial_centroids = data[random_indices]
        
        # Run K-means
        kmeans = KMeans(n_clusters=k, init=initial_centroids, n_init=1, random_state=np.random.randint(0, 1000))
        kmeans.fit(data)
        
        # Check if centroids are far enough from avoided coordinates
        centroids = kmeans.cluster_centers_
        valid = True
        
        for centroid in centroids:
            for avoid_coord in avoid_coords:
                distance = geodesic((centroid[0], centroid[1]), (avoid_coord[0], avoid_coord[1])).km
                if distance < min_distance_km:
                    valid = False
                    break
            if not valid:
                break
        
        # If valid and better than previous best, update
        if valid:
            current_cost = sum(np.min(cdist(data, centroids), axis=1))
            if current_cost < best_cost:
                best_cost = current_cost
                best_centroids = centroids
                best_labels = kmeans.labels_
    
    return best_centroids, best_labels

# Extract first best warehouse coordinates
best_warehouse_coords = best_warehouses_df[['Latitude', 'Longitude']].values

# Apply constrained K-means for second-best warehouses
second_best_centroids, second_best_labels = constrained_kmeans(
    stockist_coords, 
    k=2, 
    avoid_coords=best_warehouse_coords,
    min_distance_km=50  # Require second-best warehouses to be at least 30km from first-best
)

# If the constrained K-means didn't find valid centroids, fall back to a different approach
if second_best_centroids is None:
    logging.warning("Constrained K-means failed to find valid warehouses. Using fallback approach.")
    
    # Fallback: Divide the region into quadrants and select the best from each quadrant
    # that's not already used
    
    # Find the geographical center
    center_lat = np.mean(stockist_coords[:, 0])
    center_lon = np.mean(stockist_coords[:, 1])
    
    # Divide into quadrants (NW, NE, SW, SE) 
    quadrants = {
        'NW': january_df[(january_df['Latitude'] > center_lat) & (january_df['Longitude'] < center_lon)],
        'NE': january_df[(january_df['Latitude'] > center_lat) & (january_df['Longitude'] > center_lon)],
        'SW': january_df[(january_df['Latitude'] < center_lat) & (january_df['Longitude'] < center_lon)],
        'SE': january_df[(january_df['Latitude'] < center_lat) & (january_df['Longitude'] > center_lon)]
    }
    
    # Find most central points in the two largest quadrants
    quadrant_sizes = {k: len(v) for k, v in quadrants.items()}
    largest_quadrants = sorted(quadrant_sizes.items(), key=lambda x: x[1], reverse=True)[:2]
    
    second_best_centroids = []
    for quadrant_name, _ in largest_quadrants:
        quadrant_df = quadrants[quadrant_name]
        if len(quadrant_df) > 0:
            # Find central point in the quadrant
            quadrant_coords = quadrant_df[['Latitude', 'Longitude']].values
            quadrant_center = np.mean(quadrant_coords, axis=0)
            
            # Check if it's far enough from best warehouses
            is_far_enough = all(geodesic((quadrant_center[0], quadrant_center[1]), 
                                        (wh[0], wh[1])).km > 50 
                               for wh in best_warehouse_coords)
            
            if is_far_enough:
                second_best_centroids.append(quadrant_center)
            else:
                # Find the farthest point from any best warehouse
                max_distance = 0
                farthest_point = None
                
                for _, row in quadrant_df.iterrows():
                    min_dist_to_best = min(geodesic((row['Latitude'], row['Longitude']), 
                                                  (wh[0], wh[1])).km 
                                         for wh in best_warehouse_coords)
                    
                    if min_dist_to_best > max_distance:
                        max_distance = min_dist_to_best
                        farthest_point = [row['Latitude'], row['Longitude']]
                
                if farthest_point:
                    second_best_centroids.append(farthest_point)

    # If we still don't have 2 warehouses, add from other quadrants
    remaining_quadrants = sorted([q for q, _ in quadrant_sizes.items() 
                                 if q not in [q_name for q_name, _ in largest_quadrants]], 
                                key=lambda q: quadrant_sizes[q], reverse=True)
    
    for quadrant_name in remaining_quadrants:
        if len(second_best_centroids) >= 2:
            break
            
        quadrant_df = quadrants[quadrant_name]
        if len(quadrant_df) > 0:
            # Find the farthest point from any best warehouse
            max_distance = 0
            farthest_point = None
            
            for _, row in quadrant_df.iterrows():
                min_dist_to_best = min(geodesic((row['Latitude'], row['Longitude']), 
                                               (wh[0], wh[1])).km 
                                     for wh in best_warehouse_coords)
                
                if min_dist_to_best > max_distance:
                    max_distance = min_dist_to_best
                    farthest_point = [row['Latitude'], row['Longitude']]
            
            if farthest_point:
                second_best_centroids.append(farthest_point)
    
    second_best_centroids = np.array(second_best_centroids[:2])
    
    # Assign labels based on distance
    second_best_labels = []
    for coord in stockist_coords:
        distances = [geodesic((coord[0], coord[1]), (wh[0], wh[1])).km for wh in second_best_centroids]
        second_best_labels.append(np.argmin(distances))
    second_best_labels = np.array(second_best_labels)

# Create DataFrame for second best warehouses
second_best_warehouses = pd.DataFrame(second_best_centroids, columns=['Latitude', 'Longitude'])

# Initialize geocoder
geolocator = Nominatim(user_agent="warehouse_optimizer2")

def get_location_name(lat, lon, max_retries=3):
    """Fetch location name or nearest named town."""
    location = None
    try:
        location = geolocator.reverse((lat, lon), exactly_one=True, language="en")
        if location and location.address:
            return location.address.split(",")[0]  # Return primary name (e.g., town)
    except Exception as e:
        logging.warning(f"Reverse geocoding failed: {e}")

    # Fallback: Find nearest named stockist town within 50 km
    min_distance = float("inf")
    nearest_town = "Unnamed Location"
    for _, stockist in january_df.iterrows():
        dist = geodesic((lat, lon), (stockist["Latitude"], stockist["Longitude"])).km
        if dist < min_distance and dist <= 50:  # Search radius = 50 km
            min_distance = dist
            nearest_town = stockist["Town"]
    return f"{nearest_town} (Nearest Named Location)"

# Add location names to second best warehouses
second_best_warehouses["Location_Name"] = second_best_warehouses.apply(
    lambda row: get_location_name(row["Latitude"], row["Longitude"]), axis=1
)

# Create "assets" folder if it doesn't exist
if not os.path.exists("assets"):
    os.makedirs("assets")

# Save second best warehouses to CSV
second_best_warehouses.to_csv("second_best_warehouses.csv", index=False)

# Assign the second best warehouse to each stockist
january_df['Second_Best_Cluster'] = second_best_labels

# Calculate costs for all three scenarios
second_best_warehouse_coords = second_best_warehouses[['Latitude', 'Longitude']].values
original_cost = calculate_cost(january_df, warehouse_df[['Latitude', 'Longitude']].values)
second_best_cost = calculate_cost(january_df, second_best_warehouse_coords)
best_cost = calculate_cost(january_df, best_warehouse_coords)

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

# Generate interactive map
map_ = folium.Map(
    location=[np.mean(stockist_coords[:, 0]), np.mean(stockist_coords[:, 1])],
    zoom_start=6,
    tiles="CartoDB Positron",
    attr="© OpenStreetMap contributors | CartoDB Positron"
)

# Add stockist markers (cyan)
for _, row in january_df.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=6,
        color='cyan',
        fill=True,
        fill_color='cyan',
        fill_opacity=0.9
    ).add_to(map_)
    
# Add original warehouse markers
for idx, row in warehouse_df.iterrows():
    # Customize based on warehouse number
    if idx == 0:  # Warehouse 1
        icon_color = 'red'
        icon_type = 'home'
        prefix = 'fa'
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
            icon_color='white'
        ),
        popup=f"""
            <b>{wh_label}</b><br>
            <i>{row['Town']}</i><br>
            Coordinates: {row['Latitude']:.4f}, {row['Longitude']:.4f}
        """,
        tooltip=f"{wh_label} - {row['Town']}",
        marker_kwargs={'opacity': 0.9}
    ).add_to(map_)

# Add best warehouse markers (green)
for _, row in best_warehouses_df.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        icon=folium.Icon(color='green', icon='ok-sign'),
        popup=f"Best Warehouse: {row['Location_Name']}",
        tooltip=f"Best WH: {row['Location_Name']}"
    ).add_to(map_)

# Add second best warehouse markers (purple)
for _, row in second_best_warehouses.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        icon=folium.Icon(color='purple', icon='flag'),
        popup=f"Second Best Warehouse: {row['Location_Name']}",
        tooltip=f"2nd Best WH: {row['Location_Name']}"
    ).add_to(map_)

# Function to add routes to map
def add_routes_to_map(map_, stockist_df, warehouse_coords, color, weight, dash_array, label, cluster_col=None):
    """Add routes with custom styles to the map."""
    for i, wh in enumerate(warehouse_coords):
        if cluster_col:
            cluster_stockists = stockist_df[stockist_df[cluster_col] == i]
        else:
            # If no cluster column specified, use all stockists for each warehouse
            cluster_stockists = stockist_df
        
        for _, row in cluster_stockists.iterrows():
            stockist_loc = (row['Latitude'], row['Longitude'])
            try:
                # Get route from OpenRouteService
                route = client.directions(
                    coordinates=[(wh[1], wh[0]),  # WH coords
                               (stockist_loc[1], stockist_loc[0])],  # Stockist coords
                    profile='driving-car',
                    format='geojson'
                )
                route_points = [(point[1], point[0]) for point in 
                               route['routes'][0]['geometry']['coordinates']]
                distance = route['routes'][0]['segments'][0]['distance']/1000
            except Exception as e:
                # Fallback to straight-line distance
                route_points = [(wh[0], wh[1]), stockist_loc]
                distance = geodesic((wh[0], wh[1]), stockist_loc).km

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
add_routes_to_map(map_, w1_stockists, [w1_coords], 'red', 2, '5,5', 'Original W1')

# Add Original W2 Routes (Blue Dashed)
w2_stockists = january_df[january_df['Original_Warehouse'] == 'W2']
add_routes_to_map(map_, w2_stockists, [w2_coords], 'blue', 2, '5,5', 'Original W2')

# Add Second Best Routes (Purple Solid)
add_routes_to_map(
    map_, 
    january_df, 
    second_best_warehouse_coords, 
    'purple', 
    3, 
    None, 
    'Second Best',
    'Second_Best_Cluster'
)

# Update legend HTML
legend_html = """
<div style="position: fixed; bottom: 20px; left: 20px; z-index: 1000; 
            background: white; padding: 10px; border-radius: 5px; box-shadow: 0 0 5px rgba(0,0,0,0.2)">
    <h4>Legend</h4>
    <p style="margin:2px"><i class="fa fa-home" style="color: red"></i> Original W1</p>
    <p style="margin:2px"><i class="fa fa-industry" style="color: blue"></i> Original W2</p>
    <p style="margin:2px"><i class="fa fa-ok-sign" style="color: green"></i> Best WH</p>
    <p style="margin:2px"><i class="fa fa-flag" style="color: purple"></i> Second Best WH</p>
    <p style="margin:2px; color: cyan">● Stockists</p>
    <p style="margin:2px; color: red">--- W1 Routes</p>
    <p style="margin:2px; color: blue">--- W2 Routes</p>
    <p style="margin:2px; color: purple">➔ Second Best Routes</p>
</div>
"""
map_.get_root().html.add_child(folium.Element(legend_html))

# Save the map
map_.save("second_best_warehouse_map.html")

# Calculate individual stockist costs for all three scenarios
stockist_costs = []

for _, row in january_df.iterrows():
    stockist_location = (row['Latitude'], row['Longitude'])
    cost_per_km = TOT_df['Charge(in Rs.)'].mean()
    
    # Original warehouse distance and cost
    original_dists = [geodesic(stockist_location, (wh[0], wh[1])).km for wh in warehouse_df[['Latitude', 'Longitude']].values]
    original_dist = min(original_dists)
    original_stockist_cost = original_dist * cost_per_km
    
    # Best warehouse distance and cost
    best_dists = [geodesic(stockist_location, (wh[0], wh[1])).km for wh in best_warehouse_coords]
    best_dist = min(best_dists)
    best_stockist_cost = best_dist * cost_per_km
    
    # Second best warehouse distance and cost
    second_best_dists = [geodesic(stockist_location, (wh[0], wh[1])).km for wh in second_best_warehouse_coords]
    second_best_dist = min(second_best_dists)
    second_best_stockist_cost = second_best_dist * cost_per_km
    
    stockist_costs.append({
        'Stockist': row['Town'],
        'Original Cost (₹)': original_stockist_cost,
        'Best Cost (₹)': best_stockist_cost,
        'Second Best Cost (₹)': second_best_stockist_cost,
        'Original vs Best (₹)': original_stockist_cost - best_stockist_cost,
        'Best vs Second Best (₹)': best_stockist_cost - second_best_stockist_cost,
        'Original vs Second Best (₹)': original_stockist_cost - second_best_stockist_cost,
        'Reduction (%)': ((original_stockist_cost - second_best_stockist_cost) / original_stockist_cost * 100),
        'Status': 'Improved' if original_stockist_cost > second_best_stockist_cost else 'Loss'
    })

# Collect costs for bar chart
original_costs = [cost['Original Cost (₹)'] for cost in stockist_costs]
second_best_costs = [cost['Second Best Cost (₹)'] for cost in stockist_costs]
stockist_names = [cost['Stockist'] for cost in stockist_costs]

# Create the figure and axis for better formatted bar chart
fig, ax = plt.subplots(figsize=(14, 8))

# Bar chart for costs
width = 0.4  # Width of bars
x = np.arange(len(stockist_costs))

# Create bars with better visibility
bars1 = ax.bar(x - width/2, original_costs, width, label="Original Warehouse Cost (₹)", 
              color='#E74C3C', alpha=0.7, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, second_best_costs, width, label="Second Best Warehouse Cost (₹)", 
              color='#2ECC71', alpha=0.7, edgecolor='black', linewidth=0.5)

# Calculate percent reduction for annotation
percent_reductions = [((orig - second) / orig) * 100 if orig > 0 else 0 
                     for orig, second in zip(original_costs, second_best_costs)]

# Format bar chart axis
ax.set_xlabel("Stockists", fontsize=12, fontweight='bold')
ax.set_ylabel("Delivery Cost (₹)", fontsize=12, fontweight='bold')
ax.set_title("Comparison of Delivery Costs Before and After Optimization", fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(stockist_names, rotation=45, ha='right', fontsize=10)
ax.grid(axis='y', linestyle="--", alpha=0.5)

# Add cost reduction percentages between bars
for i, (orig, sec, pct) in enumerate(zip(original_costs, second_best_costs, percent_reductions)):
    if orig > sec:  # Only show for cost reduction
        arrow_height = (orig + sec) / 2
        ax.annotate(f"{pct:.1f}%", 
                   xy=(i, arrow_height), 
                   xytext=(i, arrow_height),
                   textcoords="data",
                   ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                   fontsize=9)

# Add cost labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.01*max(original_costs), 
           f"₹{height:.0f}", ha='center', va='bottom', fontsize=9, rotation=0)

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.01*max(original_costs), 
           f"₹{height:.0f}", ha='center', va='bottom', fontsize=9, rotation=0)

# Add total cost summary at the bottom
total_original = sum(original_costs)
total_second_best = sum(second_best_costs)
total_savings = total_original - total_second_best
percent_saving = (total_savings / total_original) * 100

summary_text = (f"Total Original Cost: ₹{total_original:,.2f}\n"
                f"Total Second Best Cost: ₹{total_second_best:,.2f}\n"
                f"Total Savings: ₹{total_savings:,.2f} ({percent_saving:.2f}%)")

# Add text box for summary
props = dict(boxstyle='round', facecolor='ivory', alpha=0.7)
ax.text(0.02, 0.02, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', bbox=props)

# Call legend with better formatting
ax.legend(
    loc="upper right",
    fontsize=10,
    frameon=True,
    edgecolor='gray',
    facecolor='white',
    shadow=True
)

plt.tight_layout()
plt.savefig("assets/secondbest.png", dpi=300, bbox_inches='tight')
plt.close()

# Create comparison dataframe
comparison_df = pd.DataFrame(stockist_costs)

# Add Total Row
total_row = pd.DataFrame({
    'Stockist': ['Total'],
    'Original Cost (₹)': [comparison_df['Original Cost (₹)'].sum()],
    'Best Cost (₹)': [comparison_df['Best Cost (₹)'].sum()],
    'Second Best Cost (₹)': [comparison_df['Second Best Cost (₹)'].sum()],
    'Original vs Best (₹)': [comparison_df['Original vs Best (₹)'].sum()],
    'Best vs Second Best (₹)': [comparison_df['Best vs Second Best (₹)'].sum()],
    'Original vs Second Best (₹)': [comparison_df['Original vs Second Best (₹)'].sum()],
    'Reduction (%)': [''],
    'Status': ['']
})

# Concatenate with the comparison dataframe
comparison_df = pd.concat([comparison_df, total_row], ignore_index=True)

# Format numbers for better readability
numeric_cols = ['Original Cost (₹)', 'Best Cost (₹)', 'Second Best Cost (₹)', 
                'Original vs Best (₹)', 'Best vs Second Best (₹)', 'Original vs Second Best (₹)']

for col in numeric_cols:
    comparison_df[col] = comparison_df[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "")

# Format percentage column
comparison_df['Reduction (%)'] = comparison_df['Reduction (%)'].apply(
    lambda x: f"{x:.2f}%" if pd.notnull(x) and x != '' else ""
)

# Save comparison to CSV
comparison_df.to_csv("Second_Best_Warehouse_Optimization_Results.csv", index=False)

# Clean up frequency columns if they exist
if all(col in january_df.columns for col in ['Frequency(W1)', 'Frequency(W2)']):
    # Clean column names by stripping extra spaces
    january_df.columns = january_df.columns.str.strip()
    
    # Drop rows with missing frequency values
    frequency_df = january_df.dropna(subset=['Frequency(W1)', 'Frequency(W2)'])
    
    if not frequency_df.empty:
        # Extract data after cleaning
        frequency_stockist_names = frequency_df['Town'].values
        w1_frequencies = frequency_df['Frequency(W1)'].values
        w2_frequencies = frequency_df['Frequency(W2)'].values
        
        # Create frequency chart
        plt.figure(figsize=(14, 7))
        
        # Line plots for delivery frequency
        plt.plot(frequency_stockist_names, w1_frequencies, color='blue', marker='o', linestyle='-', 
                linewidth=2, markersize=6, label="W1 Frequency")
        plt.plot(frequency_stockist_names, w2_frequencies, color='orange', marker='s', linestyle='--', 
                linewidth=2, markersize=6, label="W2 Frequency")
        
        # Add frequency values above points
        for i in range(len(frequency_stockist_names)):
            plt.text(i, w1_frequencies[i] + 1, f"{w1_frequencies[i]}", ha='center', fontsize=9, color='blue')
            plt.text(i, w2_frequencies[i] + 1, f"{w2_frequencies[i]}", ha='center', fontsize=9, color='orange')
        
        # Graph Formatting
        plt.xlabel("Stockists", fontsize=12)
        plt.ylabel("Delivery Frequency", fontsize=12)
        plt.title("Stockist Delivery Frequency from Original Warehouses", fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.grid(axis='y', linestyle="--", alpha=0.5)
        plt.legend(loc="upper right", fontsize=10)
        
        plt.tight_layout()
        plt.savefig('assets/Stockist_Delivery_Frequency.png', dpi=300, bbox_inches='tight')
        plt.show()

# Print summary
print(f"Original Total Cost: ₹{original_cost:,.2f}")
print(f"Best Optimization Cost: ₹{best_cost:,.2f}")
print(f"Second Best Optimization Cost: ₹{second_best_cost:,.2f}")
print(f"Savings (Original vs Best): ₹{original_cost - best_cost:,.2f}")
print(f"Savings (Original vs Second Best): ₹{original_cost - second_best_cost:,.2f}")
print(f"Difference (Best vs Second Best): ₹{second_best_cost - best_cost:,.2f}")
print("Second best warehouse optimization complete!")
print("Files generated:")
print("- second_best_warehouses.csv: Contains second best warehouse locations")
print("- second_best_warehouse_map.html: Interactive map showing all three solutions")
print("- assets/secondbest.png: Bar chart comparing costs")
print("- Second_Best_Warehouse_Optimization_Results.csv: Detailed cost comparison by stockist")
if all(col in january_df.columns for col in ['Frequency(W1)', 'Frequency(W2)']):
    if not frequency_df.empty:
        print("- assets/Stockist_Delivery_Frequency.png: Frequency chart for stockist deliveries")
