INSERT INTO public.locations (geom_id, geometry)
WITH base_points AS (
    -- Generate standard points, mapping the polygon's fid to geom_id
    SELECT 
        fid AS geom_id,
        (ST_Dump(ST_GeneratePoints(geometry, 10))).geom AS geometry
    FROM public.polygons
),
overlap_areas AS (
    SELECT 
        a.fid AS geom_id, 
        ST_Intersection(a.geometry, b.geometry) AS intersect_geom
    FROM public.polygons a
    JOIN public.polygons b 
        ON ST_Intersects(a.geometry, b.geometry) 
        AND a.fid < b.fid
),
overlap_points AS (
    -- Generate extra points in the overlaps
    SELECT 
        geom_id,
        (ST_Dump(ST_GeneratePoints(intersect_geom, 5))).geom AS geometry
    FROM overlap_areas
    WHERE ST_GeometryType(intersect_geom) IN ('ST_Polygon', 'ST_MultiPolygon')
)
-- Combine and insert
SELECT geom_id, geometry FROM base_points
UNION ALL
SELECT geom_id, geometry FROM overlap_points;