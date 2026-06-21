-- Migration: Add progress tracking fields to detection_results
-- Run this once against your existing database to add the new columns.
-- Safe to run multiple times (uses IF NOT EXISTS pattern via DO block).

DO $$
BEGIN
    -- Add status_message column for human-readable pipeline step
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'detection_results'
          AND column_name = 'status_message'
    ) THEN
        ALTER TABLE detection_results ADD COLUMN status_message VARCHAR(512);
        RAISE NOTICE 'Added column: detection_results.status_message';
    ELSE
        RAISE NOTICE 'Column already exists: detection_results.status_message';
    END IF;

    -- Add started_at column to track when processing began (for elapsed time)
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'detection_results'
          AND column_name = 'started_at'
    ) THEN
        ALTER TABLE detection_results ADD COLUMN started_at TIMESTAMP WITH TIME ZONE;
        RAISE NOTICE 'Added column: detection_results.started_at';
    ELSE
        RAISE NOTICE 'Column already exists: detection_results.started_at';
    END IF;

    -- Add geojson_b2_key and geotiff_b2_key if they don't exist yet
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'detection_results'
          AND column_name = 'geojson_b2_key'
    ) THEN
        ALTER TABLE detection_results ADD COLUMN geojson_b2_key VARCHAR(512);
        RAISE NOTICE 'Added column: detection_results.geojson_b2_key';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'detection_results'
          AND column_name = 'geotiff_b2_key'
    ) THEN
        ALTER TABLE detection_results ADD COLUMN geotiff_b2_key VARCHAR(512);
        RAISE NOTICE 'Added column: detection_results.geotiff_b2_key';
    END IF;
END
$$;
