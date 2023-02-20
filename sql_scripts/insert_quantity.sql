INSERT INTO environment_data.quantities(
	quantity_name, unit, conversion_factor
)
VALUES
	-- tmin, tmax, snow, wdir, wspd, wpgt, pres
	('AverageTemperature',	'C', 	1),
	('Precipitation',		'mm', 	1),
	('WindDirection',		'deg', 	1),
	('WindSpeed',			'km/h', 	1),
	('AirPressure',			'hPa', 	1)	;