-- ***************************************************************************
-- Loading Data:
-- create external table mapping for events.csv and mortality_events.csv
-- Note that events.csv is put under hdfs directory 'input/events/events.csv'
-- similar for mortality_events.csv
-- ***************************************************************************
-- create events table 
DROP TABLE IF EXISTS events;
CREATE EXTERNAL TABLE events (
  patient_id STRING,
  event_id STRING,
  event_description STRING,
  time DATE,
  value DOUBLE)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION 'input/events'
TBLPROPERTIES ("skip.header.line.count"="1");

-- create mortality events table 
DROP TABLE IF EXISTS mortality;
CREATE EXTERNAL TABLE mortality (
  patient_id STRING,
  time DATE,
  label INT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION 'input/mortality'
TBLPROPERTIES ("skip.header.line.count"="1");


-- ******************************************************
-- Task 1:
-- By manipulating the above two tables, 
-- generate two views for alive and dead patients' events
-- ******************************************************
-- find events for alive patients
DROP VIEW IF EXISTS alive_events;
CREATE VIEW alive_events 
AS
SELECT events.patient_id, events.event_id, events.time 
-- ***** your code below *****







-- find events for dead patients
DROP VIEW IF EXISTS dead_events;
CREATE VIEW dead_events 
AS
SELECT events.patient_id, events.event_id, events.time
-- ***** your code below *****







-- ************************************************
-- Task 2: Event count metrics
-- Compute average, min and max of event counts 
-- for alive and dead patients respectively  
-- ************************************************
-- alive patients
SELECT avg(event_count), min(event_count), max(event_count)
-- ***** your code below *****





-- dead patients
SELECT avg(event_count), min(event_count), max(event_count)
-- ***** your code below *****






-- ************************************************
-- Task 3: Encounter count metrics 
-- Compute average, min and max of encounter counts 
-- for alive and dead patients respectively
-- ************************************************
-- alive
SELECT avg(encounter_count), min(encounter_count), max(encounter_count)
-- ***** your code below *****





-- dead
SELECT avg(encounter_count), min(encounter_count), max(encounter_count)
-- ***** your code below *****







-- ************************************************
-- Task 4: Record length metrics
-- Compute average, min and max of record lengths
-- for alive and dead patients respectively
-- ************************************************
-- alive 
SELECT avg(record_length), min(record_length), max(record_length)
-- ***** your code below *****





-- dead
SELECT avg(record_length), min(record_length), max(record_length)
-- ***** your code below *****








-- ******************************************* 
-- Task 5: Common diag/lab/med
-- Compute the 5 most frequently occurring diag/lab/med
-- for alive and dead patients respectively
-- *******************************************
-- alive patients
---- diag
SELECT event_id, count(*) AS diag_count
FROM alive_events
-- ***** your code below *****


---- lab
SELECT event_id, count(*) AS lab_count
FROM alive_events
-- ***** your code below *****


---- med
SELECT event_id, count(*) AS med_count
FROM alive_events
-- ***** your code below *****




-- dead patients
---- diag
SELECT event_id, count(*) AS diag_count
FROM dead_events
-- ***** your code below *****


---- lab
SELECT event_id, count(*) AS lab_count
FROM dead_events
-- ***** your code below *****


---- med
SELECT event_id, count(*) AS med_count
FROM dead_events
-- ***** your code below *****








