-- Enable PostGIS extension for spatial capabilities
CREATE EXTENSION IF NOT EXISTS postgis;

-- Table: aois
CREATE TABLE IF NOT EXISTS aois (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    geom GEOMETRY(POLYGON, 4326) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Spatial index on aois geom
CREATE INDEX IF NOT EXISTS idx_aois_geom ON aois USING GIST (geom);

-- Table: detection_results
CREATE TABLE IF NOT EXISTS detection_results (
    id SERIAL PRIMARY KEY,
    aoi_id INTEGER NOT NULL REFERENCES aois(id) ON DELETE CASCADE,
    task_id VARCHAR(255),
    status VARCHAR(50) NOT NULL,
    change_mask_path VARCHAR(512),
    geojson_path VARCHAR(512),
    changed_area_km2 DOUBLE PRECISION,
    t1_date DATE,
    t2_date DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index on aoi_id
CREATE INDEX IF NOT EXISTS idx_detection_results_aoi_id ON detection_results (aoi_id);

-- Table: alerts
CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    result_id INTEGER NOT NULL REFERENCES detection_results(id) ON DELETE CASCADE,
    message TEXT NOT NULL,
    triggered_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_at TIMESTAMP WITH TIME ZONE
);

-- Index on result_id
CREATE INDEX IF NOT EXISTS idx_alerts_result_id ON alerts (result_id);
