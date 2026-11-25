import time
import numpy as np
import requests

URL = "http://127.0.0.1:8000/recommend/{}"
N = 1000

lat = []
for i in range(1, N + 1):
    t0 = time.perf_counter()
    r = requests.get(URL.format(i), timeout=5)
    r.raise_for_status()
    lat.append((time.perf_counter() - t0) * 1000)

lat = np.array(lat)
print("n:", N)
print("mean ms:", round(float(lat.mean()), 3))
print("P50 ms:", round(float(np.percentile(lat, 50)), 3))
print("P99 ms:", round(float(np.percentile(lat, 99)), 3))