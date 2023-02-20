-- Arnhem WaterFlow
COPY environment_data.measurements(
    measure_year,
    measure_month,
    measure_day,
    measure_value,
    quantity_id,
    location_id
) FROM
    'C:\Users\woute\Documents\GitHub\WaterEnvDataWarehouse\other_scripts\data\fixed\Arnhem_WaterFlow.csv'
WITH (FORMAT csv,
      HEADER false
      DELIMITER ',');

/*
-- Arnhem WaterLevel
COPY environment_data.measurements(
    measure_year,
    measure_month,
    measure_day,
    measure_value,
    quantity_id,
    location_id
) FROM
    'C:\Users\woute\Documents\GitHub\WaterEnvDataWarehouse\other_scripts\data\Arnhem_WaterLevel.csv'
WITH (FORMAT csv,
      HEADER false
      DELIMITER ',');

-- Arnhem AverageTemperature
COPY environment_data.measurements(
    measure_year,
    measure_month,
    measure_day,
    measure_value,
    quantity_id,
    location_id
) FROM
    'C:\Users\woute\Documents\GitHub\WaterEnvDataWarehouse\other_scripts\data\Arnhem_AverageTemperature.csv'
WITH (FORMAT csv,
      HEADER false
      DELIMITER ',');

-- Arnhem Precipitation
COPY environment_data.measurements(
    measure_year,
    measure_month,
    measure_day,
    measure_value,
    quantity_id,
    location_id
) FROM
    'C:\Users\woute\Documents\GitHub\WaterEnvDataWarehouse\other_scripts\data\Arnhem_Precipitation.csv'
WITH (FORMAT csv,
      HEADER false
      DELIMITER ',');
      
-- Arnhem WindDirection
COPY environment_data.measurements(
    measure_year,
    measure_month,
    measure_day,
    measure_value,
    quantity_id,
    location_id
) FROM
    'C:\Users\woute\Documents\GitHub\WaterEnvDataWarehouse\other_scripts\data\Arnhem_WindDirection.csv'
WITH (FORMAT csv,
      HEADER false
      DELIMITER ',');
      
-- Arnhem WindSpeed
COPY environment_data.measurements(
    measure_year,
    measure_month,
    measure_day,
    measure_value,
    quantity_id,
    location_id
) FROM
    'C:\Users\woute\Documents\GitHub\WaterEnvDataWarehouse\other_scripts\data\Arnhem_WindSpeed.csv'
WITH (FORMAT csv,
      HEADER false
      DELIMITER ',');
      
-- Arnhem AirPressure
COPY environment_data.measurements(
    measure_year,
    measure_month,
    measure_day,
    measure_value,
    quantity_id,
    location_id
) FROM
    'C:\Users\woute\Documents\GitHub\WaterEnvDataWarehouse\other_scripts\data\Arnhem_AirPressure.csv'
WITH (FORMAT csv,
      HEADER false
      DELIMITER ',');

--------------------------------------------------------------------------------------------------------------
-- Lobith WaterFlow
COPY environment_data.measurements(
    measure_year,
    measure_month,
    measure_day,
    measure_value,
    quantity_id,
    location_id
) FROM
    'C:\Users\woute\Documents\GitHub\WaterEnvDataWarehouse\other_scripts\data\Lobith_WaterFlow.csv'
WITH (FORMAT csv,
      HEADER false
      DELIMITER ',');

-- Lobith WaterLevel
COPY environment_data.measurements(
    measure_year,
    measure_month,
    measure_day,
    measure_value,
    quantity_id,
    location_id
) FROM
    'C:\Users\woute\Documents\GitHub\WaterEnvDataWarehouse\other_scripts\data\Lobith_WaterLevel.csv'
WITH (FORMAT csv,
      HEADER false
      DELIMITER ',');

-- Lobith AverageTemperature
COPY environment_data.measurements(
    measure_year,
    measure_month,
    measure_day,
    measure_value,
    quantity_id,
    location_id
) FROM
    'C:\Users\woute\Documents\GitHub\WaterEnvDataWarehouse\other_scripts\data\Lobith_AverageTemperature.csv'
WITH (FORMAT csv,
      HEADER false
      DELIMITER ',');

-- Lobith Precipitation
COPY environment_data.measurements(
    measure_year,
    measure_month,
    measure_day,
    measure_value,
    quantity_id,
    location_id
) FROM
    'C:\Users\woute\Documents\GitHub\WaterEnvDataWarehouse\other_scripts\data\Lobith_Precipitation.csv'
WITH (FORMAT csv,
      HEADER false
      DELIMITER ',');
      
-- Lobith WindDirection
COPY environment_data.measurements(
    measure_year,
    measure_month,
    measure_day,
    measure_value,
    quantity_id,
    location_id
) FROM
    'C:\Users\woute\Documents\GitHub\WaterEnvDataWarehouse\other_scripts\data\Lobith_WindDirection.csv'
WITH (FORMAT csv,
      HEADER false
      DELIMITER ',');
      
-- Lobith WindSpeed
COPY environment_data.measurements(
    measure_year,
    measure_month,
    measure_day,
    measure_value,
    quantity_id,
    location_id
) FROM
    'C:\Users\woute\Documents\GitHub\WaterEnvDataWarehouse\other_scripts\data\Lobith_WindSpeed.csv'
WITH (FORMAT csv,
      HEADER false
      DELIMITER ',');
      
-- Lobith AirPressure
COPY environment_data.measurements(
    measure_year,
    measure_month,
    measure_day,
    measure_value,
    quantity_id,
    location_id
) FROM
    'C:\Users\woute\Documents\GitHub\WaterEnvDataWarehouse\other_scripts\data\Lobith_AirPressure.csv'
WITH (FORMAT csv,
      HEADER false
      DELIMITER ',');

--------------------------------------------------------------------------------------------------------------
-- Lobith AverageTemperature
COPY environment_data.measurements(
    measure_year,
    measure_month,
    measure_day,
    measure_value,
    quantity_id,
    location_id
) FROM
    'C:\Users\woute\Documents\GitHub\WaterEnvDataWarehouse\other_scripts\data\Lobith_AverageTemperature.csv'
WITH (FORMAT csv,
      HEADER false
      DELIMITER ',');

-- Lobith Precipitation
COPY environment_data.measurements(
    measure_year,
    measure_month,
    measure_day,
    measure_value,
    quantity_id,
    location_id
) FROM
    'C:\Users\woute\Documents\GitHub\WaterEnvDataWarehouse\other_scripts\data\Lobith_Precipitation.csv'
WITH (FORMAT csv,
      HEADER false
      DELIMITER ',');
      
-- Lobith WindDirection
COPY environment_data.measurements(
    measure_year,
    measure_month,
    measure_day,
    measure_value,
    quantity_id,
    location_id
) FROM
    'C:\Users\woute\Documents\GitHub\WaterEnvDataWarehouse\other_scripts\data\Lobith_WindDirection.csv'
WITH (FORMAT csv,
      HEADER false
      DELIMITER ',');
      
-- Lobith WindSpeed
COPY environment_data.measurements(
    measure_year,
    measure_month,
    measure_day,
    measure_value,
    quantity_id,
    location_id
) FROM
    'C:\Users\woute\Documents\GitHub\WaterEnvDataWarehouse\other_scripts\data\Lobith_WindSpeed.csv'
WITH (FORMAT csv,
      HEADER false
      DELIMITER ',');
      
-- Lobith AirPressure
COPY environment_data.measurements(
    measure_year,
    measure_month,
    measure_day,
    measure_value,
    quantity_id,
    location_id
) FROM
    'C:\Users\woute\Documents\GitHub\WaterEnvDataWarehouse\other_scripts\data\Lobith_AirPressure.csv'
WITH (FORMAT csv,
      HEADER false
      DELIMITER ',');
*/