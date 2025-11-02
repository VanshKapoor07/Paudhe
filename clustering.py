import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans


# Matrix size
n = 300
x = np.arange(n)
y = np.arange(n)
xx, yy = np.meshgrid(x, y)

# Helper: Gaussian cluster generator
def gaussian_cluster(cx, cy, sigma, amp=100):
    return amp * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))

def everything(visit_no, data_points):
    x = np.array(data_points)
    print(f"Total data points: {len(data_points)}" )


    #Apply DBScan clustering

    # Run DBSCAN
    db = DBSCAN(eps=38, min_samples=2).fit(x)
    labels = db.labels_


    clusters = []



    # Print results
    unique_labels = set(labels)
    print(f"Number of clusters found: {len(unique_labels) - (1 if -1 in labels else 0)}")


    for label in unique_labels:
        if label == -1:
            print("\nNoise points:")
        else:
            print(f"\nCluster {label}:")
        cluster_points = x[labels == label]
        print(cluster_points)

        for point in cluster_points:
            clusters.append({"center": (point[0], point[1]), "sigma": (5*len(cluster_points)/12)})

    # Plot clusters
    plt.figure(figsize=(6,6))
    for label in unique_labels:
        cluster_points = x[labels == label]
        if label == -1:
            plt.scatter(cluster_points[:,0], cluster_points[:,1], c='Red', label='Noise')
        else:
            plt.scatter(cluster_points[:,0], cluster_points[:,1], label=f'Cluster {label}')

    plt.title("DBSCAN Clustering of Diseased Plants, visit no: "+str(visit_no))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()


    # Split into X and Y
    x, y = zip(*data_points)

    # Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, color='red', s=50)
    plt.title('Diseased plants Scatter Plot')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()




    # Cluster definitions (bottom-left origin, so we must flip y when creating matrix)
    # clusters = [
    #     {"center": (0, 91), "sigma": 10},
    #     {"center": (25, 25), "sigma": 10},
    #     {"center": (87, 55), "sigma": 5},
    #     {"center": (180, 20), "sigma": 10},
    #     {"center": (40, 105), "sigma": 10},
    #     {"center": (265, 50), "sigma": 10},
    #     {"center": (255, 120), "sigma": 10},
    #     {"center": (285, 190), "sigma": 4},   # very small
    #     {"center": (190, 190), "sigma": 4},   # very small
    #     {"center": (125, 275), "sigma": 6},   # small
    #     {"center": (225, 280), "sigma": 6},   # small
    # ]

    print("Clusters for matrix generation:")
    print(clusters)

    amp = 100.0
    matrix = np.zeros((n, n))

    # Flip y to match bottom-left origin system
    for c in clusters:
        cx, cy = c["center"]
        cy_flipped = n - 1 - cy
        cluster_values = gaussian_cluster(cx, cy_flipped, c["sigma"], amp)
        # matrix = np.maximum(matrix, cluster_values)
        matrix += cluster_values

    # Apply NULL regions (NaN)
    # 1. (x=0 to 75; y=150 to 225)
    y1, y2 = 150, 225
    x1, x2 = 0, 75
    # flip y for matrix indexing (top-left origin)
    matrix[n - y2:n - y1, x1:x2] = np.nan

    # 2. (x=0 to 300; y=135 to 175)
    y1, y2 = 135, 175
    x1, x2 = 0, 300
    matrix[n - y2:n - y1, x1:x2] = np.nan

    # Round values
    matrix_rounded = np.round(matrix, 2)

    # Convert to DataFrame and save CSV
    df = pd.DataFrame(matrix_rounded)
    # csv_path = "/infection_matrix_with_nulls_experimental_300x300.csv"
    # df.to_csv(csv_path, index=False, header=False)
    df.to_csv(f"infection_matrix_with_nulls_experimental_300x300_{visit_no}.csv", index=False, header=False)


    # Show preview
    display(df)

    # Summary
    summary = {
        "shape": matrix.shape,
        "nan_count": int(np.isnan(matrix).sum()),
        "nan_percent": float(np.isnan(matrix).sum()) / (n * n) * 100,
        "max": float(np.nanmax(matrix)),
        "mean_non_nan": float(np.nanmean(matrix)),
    }
    summary

    # Scale values from 0–100 to 0–255
    scaled = df * 2.55

    # Replace NaN with 0 for plotting
    scaled_filled = scaled.fillna(0)                      # Use 200 to distinguish NULL areas, keep it 0 after evry DMD iteration.

    # Convert to numpy array for plotting
    data = scaled_filled.to_numpy()

    # Plot grayscale image
    plt.figure(figsize=(6,6))
    plt.imshow(data, cmap='gray', vmin=0, vmax=255)
    plt.title("Scaled Grayscale Plot (0–100 → 0–255)")
    plt.axis('off')
    plt.show()

    return scaled_filled



data_points = [[0,10],[25,25],[51,0],[0,50],[0,58],[58,50],[87,40],[99,40],[112,40],[87,55],[150,75],[90,80],[112,80],[0,91],[20,110],[170,50],[180,20],[200,20],[225,30],[225,70],[235,70],[265,70],[265,80],[265,30],[285,30],[165,105],[175,105],[195,105],[200,105],[225,130],[225,120],[255,120],[300,150],[75,150],[190,190],[210,190],[285,190],[295,190],[115,170],[150,265],[115,250],[125,275],[135,285],[180,300],[200,275],[150,280],[255,280]]
scaled_filled1 = everything(1,data_points)

data_points = [[0,10],[25,25],[51,0],[0,50],[0,58],[58,50],[87,40],[99,40],[112,40],[87,55],[150,75],[90,80],[112,80],[0,91],[20,110],[170,50],[180,20],[200,20],[225,30],[225,70],[235,70],[265,70],[265,80],[265,30],[285,30],[165,105],[175,105],[195,105],[200,105],[225,130],[225,120],[255,120],[300,150],[75,150],[190,190],[210,190],[285,190],[295,190],[115,170],[150,265],[115,250],[125,275],[135,285],[180,300],[200,275],[150,280],[255,280],[115,265],[245,280],[225,80]]
scaled_filled2 = everything(2,data_points)

data_points = [[0,10],[25,25],[51,0],[0,50],[0,58],[58,50],[87,40],[99,40],[112,40],[87,55],[150,75],[90,80],[112,80],[0,91],[20,110],[170,50],[180,20],[200,20],[225,30],[225,70],[235,70],[265,70],[265,80],[265,30],[285,30],[165,105],[175,105],[195,105],[200,105],[225,130],[225,120],[255,120],[300,150],[75,150],[190,190],[210,190],[285,190],[295,190],[115,170],[150,265],[115,250],[125,275],[135,285],[180,300],[200,275],[150,280],[255,280],[115,265],[245,280],[225,80],[232,90],[230,70],[75,75],[80,75]]
scaled_filled3 = everything(3,data_points)

data_points = [[0,10],[25,25],[51,0],[0,50],[0,58],[58,50],[87,40],[99,40],[112,40],[87,55],[150,75],[90,80],[112,80],[0,91],[20,110],[170,50],[180,20],[200,20],[225,30],[225,70],[235,70],[265,70],[265,80],[265,30],[285,30],[165,105],[175,105],[195,105],[200,105],[225,130],[225,120],[255,120],[300,150],[75,150],[190,190],[210,190],[285,190],[295,190],[115,170],[150,265],[115,250],[125,275],[135,285],[180,300],[200,275],[150,280],[255,280],[115,265],[245,280],[225,80],[232,90],[230,70],[240,120],[245,125],[75,75],[80,75],[125,265]]
scaled_filled4 = everything(4,data_points)

data_points = [[0,10],[25,25],[51,0],[0,50],[0,58],[58,50],[87,40],[99,40],[112,40],[87,55],[150,75],[90,80],[112,80],[0,91],[20,110],[170,50],[180,20],[200,20],[225,30],[225,70],[235,70],[265,70],[265,80],[265,30],[285,30],[165,105],[175,105],[195,105],[200,105],[225,130],[225,120],[255,120],[300,150],[75,150],[190,190],[210,190],[285,190],[295,190],[115,170],[150,265],[115,250],[125,275],[135,285],[180,300],[200,275],[150,280],[255,280],[115,265],[245,280],[225,80],[232,90],[230,70],[240,120],[245,125],[255,125],[75,75],[80,75],[125,265],[180,150]]
scaled_filled5 = everything(5,data_points)


if __name__ == "__main__":
    pass