File naming convention:
-----------------------

The file name is based on the subject number(N) and trial number(T).

For every subject there are multiple trials. Each trial has two files associated
with it: a data file and an index file.

The file naming convention is

s#t$data.daq
s#t$index.mat

where # indicates the subject number and $ indicates the trial number

For example:

For subject 4, 2:
s4t2data.daq
s4t2index.mat

-----------------------------------------------------------------------------

s#t$data.daq

This is a myoelectric data file. Use a matlab function 'daqread' to read the data 
from this daq file. This file contains eight channels of myoelectric data. Refer
to the 'electrode placments.jpg' to determine the electrode sites.

-----------------------------------------------------------------------------

s#t$index.mat

The index file has two variables: motion and start_index

The variable 'motion', represents the order in which the seven forearm motions
(hand open, hand close, wrist flexion, wrist extension, supination, pronation)
were performed by the subject. 
 
The motions are numbered from 1 to 7 as follows: 

1 Hand Open
2 Hand Close
3 Wrist Flexion
4 Wrist Extension
5 Supination
6 Pronation
7 Rest

Each motion is performed four times throughout a trial.

Please note that every trial starts and ends with additional rest periods, as 
indicated in the motion array. 
