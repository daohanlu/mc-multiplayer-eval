#!/usr/bin/env python3
"""Extract GCS paths from experiment CSV and generate download commands."""

import csv
import re
from urllib.parse import urlparse

# Read the CSV file
csv_file = "Solaris Experiments - Multiplayer Final Experiments.csv"

# Column names to extract
output_columns = [
    "Translation Output",
    "Rotation Output",
    "One looks away Output",
    "Both look away Output",
    "Building Output"
]

# Parse CSV
with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)

    for row in reader:
        experiment_name = row.get('Name', '').strip()

        # Skip empty rows or rows without experiment names
        if not experiment_name:
            continue

        print(f"\n# Experiment: {experiment_name}")
        # print mkdir command
        print(f"mkdir -p ./generations/{experiment_name}")

        for col in output_columns:
            url = row.get(col, '').strip()

            if not url:
                continue

            # Extract GCS path from console URL
            # Format: https://console.cloud.google.com/storage/browser/bucket/path/to/data?...
            match = re.search(r'storage/browser/([^?]+)', url)

            if match:
                path_parts = match.group(1).split('/')
                bucket = path_parts[0]
                gcs_path = '/'.join(path_parts[1:])

                # Clean up any URL-encoded characters and remove trailing parameters
                gcs_path = gcs_path.split(';')[0]  # Remove ;tab=objects etc

                gs_url = f"gs://{bucket}/{gcs_path}"

                # Generate download command
                eval_type = col.replace(" Output", "").lower().replace(" ", "_")
                output_dir = f"./generations/{experiment_name}"

                print(f"gcloud storage cp -r {gs_url} {output_dir}/")
