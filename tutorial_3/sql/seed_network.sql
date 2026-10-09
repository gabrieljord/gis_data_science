-- seed_network.sql
CREATE EXTENSION IF NOT EXISTS postgis;

DROP TABLE IF EXISTS spans CASCADE;
DROP TABLE IF EXISTS nodes CASCADE;
DROP TABLE IF EXISTS cables CASCADE;
DROP TABLE IF EXISTS equipment CASCADE;
DROP TABLE IF EXISTS vaults CASCADE;
DROP TABLE IF EXISTS premises CASCADE;

-- 1. Simple Topology Tables
CREATE TABLE nodes (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50),
    geom GEOMETRY(Point, 4326) NOT NULL
);

CREATE TABLE spans (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50),
    geom GEOMETRY(LineString, 4326) NOT NULL
);

-- 2. Hierarchical Network Tables
CREATE TABLE vaults (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50),
    geom GEOMETRY(Polygon, 4326)
);

CREATE TABLE equipment (
    id SERIAL PRIMARY KEY,
    type VARCHAR(50),
    max_ports INTEGER,
    geom GEOMETRY(Point, 4326)
);

CREATE TABLE premises (
    id SERIAL PRIMARY KEY,
    address VARCHAR(100),
    geom GEOMETRY(Point, 4326)
);

CREATE TABLE cables (
    id SERIAL PRIMARY KEY,
    type VARCHAR(50),
    geom GEOMETRY(LineString, 4326)
);

-- FIXTURES

-- Nodes
INSERT INTO nodes (name, geom) VALUES
('Pole_A', ST_SetSRID(ST_MakePoint(-111.9400, 33.4255), 4326)),
('Pole_B', ST_SetSRID(ST_MakePoint(-111.9300, 33.4255), 4326));

-- Spans (Valid, Dangling, and Micro-segment)
INSERT INTO spans (name, geom) VALUES
('Valid_Span', ST_SetSRID(ST_MakeLine(ST_MakePoint(-111.9400, 33.4255), ST_MakePoint(-111.9300, 33.4255)), 4326)),
('Dangling_Span', ST_SetSRID(ST_MakeLine(ST_MakePoint(-111.9400, 33.4255), ST_MakePoint(-111.9350, 33.4200)), 4326)),
('Micro_Span', ST_SetSRID(ST_MakeLine(ST_MakePoint(-111.940000, 33.425500), ST_MakePoint(-111.940001, 33.425501)), 4326));

-- Vaults & Equipment
INSERT INTO vaults (name, geom) VALUES 
('Handhole_1', ST_Buffer(ST_SetSRID(ST_MakePoint(-111.94, 33.42), 4326), 0.0001));

INSERT INTO equipment (id, type, max_ports, geom) VALUES 
(1, 'SPLITTER', 4, ST_SetSRID(ST_MakePoint(-111.94, 33.42), 4326)),
(2, 'SPLITTER', 8, ST_SetSRID(ST_MakePoint(-111.95, 33.43), 4326)); -- Invalid: floating outside vault

-- Premises & Cables
INSERT INTO premises (id, geom) VALUES 
(101, ST_SetSRID(ST_MakePoint(-111.941, 33.421), 4326));

INSERT INTO cables (id, type, geom) VALUES 
(1001, 'DROP', ST_MakeLine(ST_SetSRID(ST_MakePoint(-111.94, 33.42), 4326), ST_SetSRID(ST_MakePoint(-111.941, 33.421), 4326))),
(1002, 'FEEDER', ST_MakeLine(ST_SetSRID(ST_MakePoint(-111.94, 33.42), 4326), ST_SetSRID(ST_MakePoint(-111.942, 33.422), 4326)));
