-- ***************************************************************************
-- Aggregate events into features of patient and generate training, testing data
-- for mortality prediction. Steps have been provided to guide you. 
-- You can include as many intermediate steps as required to complete the calculations.
-- ***************************************************************************

-- register a python UDF for converting data into SVMLight format
REGISTER utils.py USING jython AS utils;

-- load events file 
events = LOAD '../../data/train/events.csv' USING PigStorage(',') AS (patientid:chararray, eventid:chararray, eventdesc:chararray, timestamp:chararray, value:float);

-- select required columns from events
events = FOREACH events GENERATE patientid, eventid, ToDate(timestamp, 'yyyy-MM-dd') AS etimestamp, value;

-- load mortality files
mortality = LOAD '../../data/train/mortality.csv' USING PigStorage(',') as (patientid:chararray, timestamp:chararray, label:int);

mortality = FOREACH mortality GENERATE patientid, ToDate(timestamp, 'yyyy-MM-dd') AS mtimestamp, label;

--DISPLAY mortality (can remove this statement)
DUMP mortality;

-- ***************************************************************************
-- Compute the index dates for dead and alive patients
-- ***************************************************************************
eventswithmort = -- perform join of events and mortality by patientid;

deadevents = -- filter and detect the events of dead patients and create it of the form (patientid, eventid, value, label, time_difference) where time_difference is the days between death date and each event timestamp

aliveevents = -- filter and detect the events of alive patients and create it of the form (patientid, eventid, value, label, time_difference) where time_difference is the days between index date and each event timestamp


-- ***************************************************************************
-- Filter events within the observation window and remove events with missing values
-- ***************************************************************************
filtered = -- contains only events for all patients within an observation window of 1000 days and is of the form (patientid, eventid, value, label, time_difference)


-- ***************************************************************************
-- Aggregate events to create features
-- ***************************************************************************

featureswithid = -- for group of (patientid, eventid), count the number of lab, medication and diagnosis events occurred for the patient and create it (patientid, 'eventid_COUNT' AS featureid, featurevalue)

-- ***************************************************************************
-- Generate feature mapping
-- ***************************************************************************
all_features = -- compute the set of distinct featureids obtained from previous step and rank features by featureid to create (idx, featureid)

-- store the features as output file. The count obtained in the last row from this file will be used to determine input parameter f in train.py
STORE all_features INTO 'features' using PigStorage(' ');

features = -- perform join of featureswithid and all_features by featureid and replace featureid with idx. It is of the form (patientid, idx, featurevalue)

-- ***************************************************************************
-- Normalize the values using min-max normalization
-- ***************************************************************************
maxvalues = -- group events by idx and compute the maximum feature value in each group, is of the form (idx, maxvalues)

normalized = -- join features and maxvalues by idx

features = -- compute the final set of normalized features of the form (patientid, idx, normalizedfeaturevalue)


-- ***************************************************************************
-- Generate features in svmlight format 
-- features is of the form (patientid, idx, normalizedfeaturevalue) and is the output of the previous step
-- e.g.  1,1,1.0
--  	 1,3,0.8
--	     2,1,0.5
--       3,3,1.0
-- ***************************************************************************
grpd = GROUP features BY patientid;

-- aggregate features of same patient together in sparse format
grpd = GROUP features BY patientid;
features = FOREACH grpd {
    sorted = ORDER features BY featureid;
    generate group as patientid, utils.bag_to_svmlight(sorted) as sparsefeature;
}

-- ***************************************************************************
-- Split into train and test set
-- labels is of the form (patientid, label) and contains all patientids followed by label of 1 for dead and 0 for alive 
-- e.g. 1,1
--	    2,0
--      3,1
-- ***************************************************************************

labels = -- create it of the form (patientid, label)

-- randomly split data for training and testing
samples = JOIN labels BY patientid, features BY patientid;
samples = FOREACH samples GENERATE $1 AS label, $3 AS sparsefeature;
samples = FOREACH samples GENERATE RANDOM() as assignmentkey, *;
SPLIT samples INTO testing IF assignmentkey <= 0.20, training OTHERWISE;
training = FOREACH training GENERATE $1..;
testing = FOREACH testing GENERATE $1..;

-- save training and tesing data
STORE testing INTO 'testing' USING PigStorage(' ');
STORE training INTO 'training' USING PigStorage(' ');