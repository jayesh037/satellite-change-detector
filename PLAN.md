This is my project architecture

🧠 FULL PROJECT ARCHITECTURE (END-TO-END)

🔷 Core Idea

Detect changes between satellite images over time and alert users for meaningful (human-made) changes.

🏗️ OVERALL SYSTEM ARCHITECTURE

User (Frontend)

&#x20;     ↓

Backend API (FastAPI)

&#x20;     ↓

\-----------------------------------

| Data Layer | ML Layer | Alert Layer |

\-----------------------------------

&#x20;     ↓

GIS Output + Visualization

🔄 COMPLETE PIPELINE (REAL FLOW)

This is your actual working pipeline 👇



1\. Dataset →

2\. Preprocessing →

3\. Model Training →

4\. Inference →

5\. Post-processing →

6\. GIS Output →

7\. Alert System →

8\. Web Visualization

👉 This follows real-world satellite pipelines where:



Data → preprocessing → comparison → change map → analysis (ResearchGate)

🧩 PHASE 1: MODEL DEVELOPMENT (START HERE)

🎯 Goal:

Train Siamese UNet

📦 Step 1: Dataset

Use: LEVIR-CD

Input:

t1 image

t2 image

label

⚙️ Step 2: Preprocessing

You will do:

Resize images (512 → 256 patches)

Normalize pixel values

(Later) NDVI + cloud masking

👉 Preprocessing improves accuracy significantly (MDPI)

🧠 Step 3: Model (CORE)

👉 Siamese UNet



Flow:

Image\_t1 → Encoder

Image\_t2 → Encoder

&#x20;       ↓

Feature comparison

&#x20;       ↓

Decoder

&#x20;       ↓

Change mask

👉 Siamese models compare two images effectively for change detection (IRJET)

⚙️ Step 4: Training

Loss: BCE + Dice

Optimizer: Adam

Output: binary mask

📊 Step 5: Evaluation

IoU

F1 Score

🧩 PHASE 2: REAL-WORLD PIPELINE (IMPORTANT)

🎯 Goal:

Make your system usable on real satellite data

🛰️ Step 6: Data Ingestion

👉 Use:



Bhoonidhi satellite images

🧰 Step 7: Raster Processing (Rasterio / GDAL)

You will:

Read GeoTIFF

Extract bands (RGB + NIR)

Align images

👉 Satellite pipelines require calibration + correction before analysis (arXiv)

🧩 PHASE 3: INFERENCE PIPELINE

🔍 Step 8: Run Model

Input:



t1 + t2 images

Output:



Change mask

🧹 Step 9: Post-processing (CRITICAL)

You will:

Remove noise

Filter small regions

NDVI filtering

👉 Goal:



Remove seasonal changes

Keep real changes

🧩 PHASE 4: GIS OUTPUT

🗺️ Step 10: Convert Output

Convert:



Mask → Polygons

Output:



GeoJSON

Shapefile

GeoTIFF

🧩 PHASE 5: BACKEND SYSTEM

🔌 FastAPI Backend

Handles:

AOI storage

Model execution

Data retrieval

Alert triggering

🗄️ Database

PostgreSQL + PostGIS

Stores:

AOIs

Results

🧩 PHASE 6: ALERT SYSTEM 🚨

Logic:

IF change\_area > threshold

→ Send alert

Methods:

Email

Dashboard

🧩 PHASE 7: FRONTEND (WEB APP)

You will build:

Features:

Map (AOI selection)

Change visualization

Timeline view

Alerts panel

🚀 ADVANCEMENT (MAKE YOUR PROJECT STAND OUT)

Once base system works 👇

🔥 Level 1 Upgrade

Add NDVI filtering

Add cloud masking

🔥 Level 2 Upgrade

Classify type of change:

Building

Vegetation

Water

🔥 Level 3 Upgrade

Replace model:

ChangeFormer

STANet

🔥 Level 4 (Advanced Research)

Real-time monitoring

Graph Neural Networks

Multi-temporal (more than 2 images)

🧠 FINAL COMPLETE FLOW (SIMPLIFIED)

User selects AOI

&#x20;       ↓

Fetch satellite images

&#x20;       ↓

Preprocess (clean data)

&#x20;       ↓

Run Siamese UNet

&#x20;       ↓

Get change map

&#x20;       ↓

Filter noise + vegetation

&#x20;       ↓

Convert to GIS format

&#x20;       ↓

Store + visualize

&#x20;       ↓

Send alerts 🚨

🎯 WHAT YOU WILL DO FIRST vs LATER

🟢 FIRST (Must do)

Train Siamese UNet on LEVIR-CD

Get change mask output

🟡 SECOND

Add preprocessing (NDVI, filtering)

Improve results

🟠 THIRD

Use Bhoonidhi data

Use Rasterio

🔴 FINAL

Build full system (frontend + backend + alerts)

💡 Golden Insight

👉 Your project is NOT just ML

It is:



30% Model

30% Data processing

40% System design

