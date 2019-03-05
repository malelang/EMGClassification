win_size = 256;
win_inc = 128; % training data has 50% overlap between windows

% create training set
[training_data,training_motion,training_index] = load_data('s4t1');
feature_training = extract_feature(training_data,win_size,win_inc);
class_training = getclass(training_data,training_motion,training_index,win_size,win_inc);

% uncomment this code to include two sets of data in the training set
% [training_data,training_motion,training_index] = load_data('s4t2');
% feature_training_tmp = extract_feature(training_data,win_size,win_inc);
% class_training_tmp = getclass(training_data,training_motion,training_index,win_size,win_inc);
% 
% feature_training = [feature_training; feature_training_tmp];
% class_training = [class_training; class_training_tmp];

[feature_training,class_training] = remove_transitions(feature_training,class_training);

% create testing set (uncomment which trial you want to test on
% [testing_data,testing_motion,testing_index] = load_data('s4t1');
% [testing_data,testing_motion,testing_index] = load_data('s4t2');
[testing_data,testing_motion,testing_index] = load_data('s4t3');
% [testing_data,testing_motion,testing_index] = load_data('s4t4');
% [testing_data,testing_motion,testing_index] = load_data('s4t5');
% [testing_data,testing_motion,testing_index] = load_data('s4t6');

win_inc = 32; % testing data has 87.5% overlap between windows
feature_testing = extract_feature(testing_data,win_size,win_inc);
class_testing = getclass(testing_data,testing_motion,testing_index,win_size,win_inc);

% leave commented to have to feature reduction
% uncomment one of these lines for either PCA feature reduction
% or ULDA feature reduction
Nfeat = 10; % number of features to reduce to
%[feature_training,feature_testing] = pca_feature_reduction(feature_training,Nfeat,feature_testing);
%[feature_training,feature_testing] = ulda_feature_reduction(feature_training,Nfeat,class_training,feature_testing);

% no post-processing
[error_training,error_testing,classification_training,classification_testing]...
    = ldaclassify(feature_training,feature_testing,class_training,class_testing);

% majority vote smoothing
classification_testing_maj = majority_vote(classification_testing,8,0);
error_testing_maj = sum(classification_testing_maj ~= class_testing)/length(class_testing)*100;

% remove transitions from computation of classification accuracy
[classification_testing_nt,class_testing_nt] = remove_transitions(classification_testing,class_testing);
error_testing_nt = sum(classification_testing_nt ~= class_testing_nt)/length(class_testing_nt)*100;

% majority vote smooth and remove transitions from computation of classification accuracy
[classification_testing_maj_nt,class_testing_nt] = remove_transitions(classification_testing_maj,class_testing);
error_testing_maj_nt = sum(classification_testing_maj_nt ~= class_testing_nt)/length(class_testing_nt)*100;

figure(1)
subplot(2,2,1)
classification_timeplot(class_testing,classification_testing);
title(['Error = ' num2str(error_testing) '%'])
subplot(2,2,2)
classification_timeplot(class_testing,classification_testing_maj);
title(['Majority Vote Error = ' num2str(error_testing_maj) '%'])
subplot(2,2,3)
classification_timeplot(class_testing_nt,classification_testing_nt);
title(['No Transitions Error = ' num2str(error_testing_nt) '%'])
subplot(2,2,4)
classification_timeplot(class_testing_nt,classification_testing_maj_nt);
title(['Majority Vote/No Transitions Error = ' num2str(error_testing_maj_nt) '%'])

figure(2)
subplot(2,2,1)
confusion_matrix = confmat(class_testing,classification_testing);
plotconfmat(confusion_matrix);
title(['Error = ' num2str(error_testing) '%'])
subplot(2,2,2)
confusion_matrix = confmat(class_testing,classification_testing_maj);
plotconfmat(confusion_matrix);
title(['Majority Vote Error = ' num2str(error_testing_maj) '%'])
subplot(2,2,3)
confusion_matrix = confmat(class_testing_nt,classification_testing_nt);
plotconfmat(confusion_matrix);
title(['No Transitions Error = ' num2str(error_testing_nt) '%'])
subplot(2,2,4)
confusion_matrix = confmat(class_testing_nt,classification_testing_maj_nt);
plotconfmat(confusion_matrix);
title(['Majority Vote/No Transitions Error = ' num2str(error_testing_maj_nt) '%'])
