
win_size = 256;
win_inc = 128; % training data has 50% overlap between windows
% DATOS DE ENTRENAMIENTO
[training_data,training_motion,training_index] = load_data('s4t1');
feature_training = extract_feature(training_data,win_size,win_inc);
class_training = getclass(training_data,training_motion,training_index,win_size,win_inc);
[feature_training,class_training] = remove_transitions(feature_training,class_training);

% DATOS DE PRUEBA
[testing_data,testing_motion,testing_index] = load_data('s4t3');
win_inc = 32; % testing data has 87.5% overlap between windows
feature_testing = extract_feature(testing_data,win_size,win_inc);
class_testing = getclass(testing_data,testing_motion,testing_index,win_size,win_inc);
%% SEGUNDO CLASIFICADOR
addpath('C:\MATLAB2018\MATLAB\mcode\BioMedicine\prtools')
[Data_training,PC_training,Ws_training,W_training] = clasificadorEMG(feature_training,class_training);
%%
[Data_testing,PC_testing,Ws_testing,W_testing] = clasificadorEMG(feature_testing,class_testing);
disp('Cargando clasificacion..')
%%
LABEL = labeld(Data_testing,Ws_training*PC_training*W_training{2})  