win_size = 256;
win_inc = 128; % training data has 50% overlap between windows

% create training set
[training_data,training_motion,training_index] = load_data('s4t1');
feature_training = extract_feature(training_data,win_size,win_inc);

