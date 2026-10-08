CREATE TABLE redrawn_polygons AS
WITH ranked AS (
    SELECT
        p.fid,
        p.geom_id,
        p.geometry AS geom,
        COALESCE(pt.point_count, 0) AS priority
    FROM public.polygons p
    LEFT JOIN (
        SELECT pol.fid, COUNT(*) AS point_count
        FROM public.locations l
        JOIN public.polygons pol
            -- Fixed typo here: changed sl.geometry to l.geometry
            ON ST_Within(l.geometry, pol.geometry) 
            AND l.geom_id = pol.geom_id
        GROUP BY pol.fid
    ) pt ON pt.fid = p.fid
)
SELECT
    r.fid,
    r.geom_id,
    ST_Multi(
        ST_CollectionExtract(
            ST_MakeValid(
                CASE
                    WHEN higher.geom_union IS NULL THEN r.geom
                    ELSE ST_Difference(r.geom, higher.geom_union)
                END
            ), 3
        )
    ) AS wkb_geometry
FROM ranked r
LEFT JOIN LATERAL (
    SELECT ST_Union(o.geom) AS geom_union
    FROM ranked o
    WHERE o.fid <> r.fid
      AND ST_Intersects(r.geom, o.geom)
      AND NOT ST_Touches(r.geom, o.geom)          
      AND (o.priority > r.priority
           OR (o.priority = r.priority AND o.fid < r.fid))
) higher ON true;