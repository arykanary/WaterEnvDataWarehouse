-- https://stackoverflow.com/questions/53770816/postgres-creating-database-using-sql-file
-- https://www.postgresqltutorial.com/postgresql-administration/postgresql-create-schema/
-- https://hasura.io/learn/database/postgresql/core-concepts/6-postgresql-relationships/
-- https://www.postgresql.org/docs/current/datatype.html

CREATE SCHEMA environment_data
	-- 
	CREATE TABLE quantities (
		-- any quantity has a name and unit
		quantity_id SMALLSERIAL PRIMARY KEY,
		name TEXT NOT NULL,
		unit TEXT NOT NULL,
		
		-- optionally a reference (eq NAP)
		reference TEXT
	)

	CREATE TABLE locations (
		--
		location_id SMALLSERIAL PRIMARY KEY,
		name TEXT NOT NULL,
		exact_location BOOLEAN NOT NULL,
		lattitude REAL NOT NULL,
		longitude REAL NOT NULL,
		
		-- if this is an RWS location it has a code, x & y
		rws_code TEXT,
		rws_x REAL,
		rws_y REAL,
		
		-- if this is an OWM location it has a code
		owm_code TEXT
	)
	
	CREATE TABLE measurement (
		-- Any measurement mean is measured on a date and is referenced to quantity and location
		measurement_id BIGSERIAL PRIMARY KEY,
		measure_date DATE NOT NULL,
		measure_value REAL NOT NULL,
		
		-- quantity and location are saved in their respective tables
		quantity_id SMALLINT NOT NULL,
		location_id SMALLINT NOT NULL,

		CONSTRAINT fk_quantity FOREIGN KEY(quantity_id) REFERENCES quantities(quantity_id),
		CONSTRAINT fk_location FOREIGN KEY(location_id) REFERENCES locations(location_id)

	)
