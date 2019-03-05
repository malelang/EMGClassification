
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
 %% EXPORTACION DE CONJUNTO DE DATOS
% A = prdataset (DATA, LABELS)
%   DATA : Tamaño [M,K], M datavectores de tamaño K 
%   LABELS: Tamaño [M,N], Array para M datavectores
%   featlab: size [K,F] array with labels for the K features%% NORMALIZACION
%Dataset = normc(training_data);
lista_feasures = char('Vrms_E1','Vrms_E2','Vrms_E3','Vrms_E4','Vrms_E5','Vrms_E6','Vrms_E7','Vrms_E8','Coef9','Coef10','Coef11','Coef12','Coef13','Coef14','Coef15','Coef16','Coef17','Coef18','Coef19','Coef20','Coef21','Coef22','Coef23','Coef24','Coef25','Coef26','Coef27','Coef28','Coef29','Coef30','Coef31','Coef32','Coef33','Coef34','Coef35','Coef36','Coef37','Coef38','Coef39','Coef40');
DatasetTotal = prdataset(feature_training, class_training,'featlab',lista_feasures);

%% SELECCION DE CARACTERISTICAS
%   featselm(TrainingSet,Criterium, Method, Number of Feasures)
%   Method NN : DEFAULT
% 	Ws: Feature selection mapping
%   R: Matrix with step by step results

 No_feasures = 40;                                        % Atributos Seleccionados
 [Ws,Rs] = featselm(DatasetTotal,'NN','ind',No_feasures); % Feasure Selection map
 DatasetTotal = DatasetTotal*Ws;                          % Map According to Attibutes selected
 DatasetTotal.featlab
 %% SELECCION DATOS DE ENTRENAMIENTO Y DE PRUEBA
%   [A,B,IA,IB] = gendat(X,N,SEED)
%   X:DATASET
%   N:Fila de elementos de cada clase
%   A, B: DATASETS
%   IA,IB: Indices originales

randreset;                                               % Para que la semilla del generador aleatorio de datos sea siempre la misma 
[Ae,Ap,IL,IU] = gendat(DatasetTotal,0.75);%% 
figure
scatterd(DatasetTotal);
%% ANALISIS DE COMPONENTES PPALES
PC = pcam(Ae,0.99999);
figure
scatterd(DatasetTotal*PC);   
        
%% ENTRENAMIENTO DE LOS CLASIFICADORES
w1 = nmc(Ae*PC); 
w2 = knnc(Ae*PC,3);
w3 = ldc(Ae*PC);        %linear
w4 = qdc(Ae*PC);        % quadratic
w5 = parzenc(Ae*PC);    % Parzen
w6 = dtc(Ae*PC);        % decision tree
w7 = svc(Ae*PC,proxm('p',6));
w8 = neurc(Ae*PC);      % Neuronal
%w9 = adaboostc(Ae*PC,perlc([],1),50,[],1); %AdaBoost

% Compute and display errors
% Store classifiers in a cell
W = {w1,w2,w3,w4,w5,w6,w7,w8};
name_classf = {'nmc','knnc','ld','qdc','parzenc','dtc','svc','neurc'};
[E,C]=testc(Ap*PC*W); % E, error, C{i} # errores de clasificaciòn en cada clase. Ver tambien confmat  
disp('Error de los clasificadores')
disp(E)
disp('Error minimo')
minE = min(E);
disp(minE)
IminE = find(E==minE);
disp('Mejores Clasificadores')  % Nombres de los clasificadores con mìnimo error de clasificaciòn
disp(name_classf(IminE))
mejorclasificador=name_classf(IminE);
