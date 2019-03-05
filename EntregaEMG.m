%% CLASIFICACIÓN DE MOVIMIENTOS EN SEÑALES EMG
% En este script se hace la clasificación de 7 tipos de movimientos, para
% los cuales se han posicionado 8 electrodos. En la primera parte
% identificaremos el dataset de entrenamiento y sacaremos las
% características y lista de clases las cuales usaremos para alimentar el
% modelo de entrenamiento en la herramienta prtools. Posteriormente
% usaremos cualquier otra de las señales como testing.

%% PARTE I: CARGA DE LOS DATOS
% En esta parte se hacen los siguientes procedimientos:
% 1. Carga los datos en el directorio data
% 2. Filtra a pasabanda de 10-400Hz
% 3. Submuestrea de 3000 a 1000Hz.
% 4. Corta el resto de información del inicio y final de los archivos

win_size = 256;
win_inc = 128; % El solapamiento de la ventana es del 50% en entrenamiento
[training_data,training_motion,training_index] = load_data('s4t1');

%% PARTE II: EXTRACCIÓN DE LAS CARACTERISTICAS
% En esta parte se extraen las características de la señal:
% 1. RMS
% 2. Coeficientes de autoregresión
% 3. Valor absoluto medio
% 4. Cruces por cero
% 5. Cambios de pendiente

feature_training = extract_feature(training_data,win_size,win_inc);

%% PARTE III: EXTRACCIÓN DE LAS CLASES
% En esta parte obtenemos un vector el cual nos indica qué movimiento está
% haciendo en cada ventana de 128 ms, esto nos va a servir para alimentar
% al clasificador en prtools.

class_training = getclass(training_data,training_motion,training_index,win_size,win_inc);

%% PARTE IV: REMOCIÓN DE TRANSICIONES
% En esta parte removemos simplemente los cambios de una actividad a otra.

[feature_training,class_training] = remove_transitions(feature_training,class_training);

%% PARTE V: CREACIÓN DEL DATASET Y ENTRENAMIENTO DEL CLASIFICADOR

% Adicionamos el path para poder trabajar con prtools
addpath('/Users/alejandralandinez/Documents/MATLAB/prtools/prtools')

%Creamos la lista de características
lista_feasures = char('Vrms_E1','Vrms_E2','Vrms_E3','Vrms_E4','Vrms_E5','Vrms_E6','Vrms_E7','Vrms_E8','Coef9','Coef10','Coef11','Coef12','Coef13','Coef14','Coef15','Coef16','Coef17','Coef18','Coef19','Coef20','Coef21','Coef22','Coef23','Coef24','Coef25','Coef26','Coef27','Coef28','Coef29','Coef30','Coef31','Coef32','Coef33','Coef34','Coef35','Coef36','Coef37','Coef38','Coef39','Coef40','mav1','mav2','mav3','mav4','mav5','mav6','mav7','mav8','zc1','zc2','zc3','zc4','zc5','zc6','zc7','zc8','ssc1','ssc2','ssc3','ssc4','ssc5','ssc6','ssc7','ssc8');

%Creación del dataset
DatasetTotal = prdataset(feature_training, class_training,'featlab',lista_feasures);

%Selección de características

 No_feasures = 64;                                        % Atributos Seleccionados
 [Ws,Rs] = featselm(DatasetTotal,'NN','ind',No_feasures); % Feasure Selection map
 DatasetTotal = DatasetTotal*Ws;                          % Map According to Attibutes selected
 DatasetTotal.featlab
 
%% PARTE VI: SELECCION DATOS DE ENTRENAMIENTO Y DE PRUEBA
%   [A,B,IA,IB] = gendat(X,N,SEED)
%   X:DATASET
%   N:Fila de elementos de cada clase
%   A, B: DATASETS
%   IA,IB: Indices originales

randreset;% Para que la semilla del generador aleatorio de datos sea siempre la misma 
[Ae,Ap,IL,IU] = gendat(DatasetTotal,0.75); 
figure
scatterd(DatasetTotal);

% ANALISIS DE COMPONENTES PPALES
PC = pcam(Ae,0.99999);
figure
scatterd(DatasetTotal*PC);   
        
%% PARTE VII: ENTRENAMIENTO DE LOS CLASIFICADORES
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
[E,C]=testc(Ap*PC*W); % E, error, C{i} # errores de clasificaci�n en cada clase. Ver tambien confmat  
disp('Error de los clasificadores')
disp(E)
disp('Error minimo')
minE = min(E);
disp(minE)
IminE = find(E==minE);
disp('Mejores Clasificadores')  % Nombres de los clasificadores con m�nimo error de clasificaci�n
disp(name_classf(IminE))
mejorclasificador=name_classf(IminE);
