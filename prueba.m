%% CLASIFICACIÓN DE LOS DATOS DE PRUEBA
% En este script se hace la clasificación de 7 tipos de movimientos, para
% los cuales se han posicionado 8 electrodos. En la primera parte
% identificaremos el dataset de entrenamiento y sacaremos las
% características y lista de clases las cuales usaremos para alimentar el
% modelo de entrenamiento en la herramienta prtools. Posteriormente
% usaremos cualquier otra de las señales como testing.

%% I PARTE: EXTRACCIÓN DE LOS DATOS DE ENTRENAMIENTO 
%Declaramos el tamaño de la ventana
win_size = 256;
win_inc = 128; % El solapamiento de la ventana es del 50% en entrenamiento
%Cargamos los datos de entrenamiento
[training_data,training_motion,training_index] = load_data('s4t1');
%Extraemos las características
feature_training = extract_feature(training_data,win_size,win_inc);
%Extraemos las clases
class_training = getclass(training_data,training_motion,training_index,win_size,win_inc);
%Removemos las transiciones
[feature_training,class_training] = remove_transitions(feature_training,class_training);

%% II PARTE: EXTRACCIÓN DE LOS DATOS DE PRUEBA
%Declaramos el tamaño de la ventana
win_inctesting = 32; % El solapamiento de la ventana es del 82.5% en entrenamiento
%Cargamos los datos de entrenamiento
[testing_data,testing_motion,testing_index] = load_data('s4t3');
%Extraemos las características
feature_testing = extract_feature(testing_data,win_size,win_inctesting);
%Extraemos las clases
class_testing = getclass(testing_data,testing_motion,testing_index,win_size,win_inctesting);

%% III PARTE: ENTRENAMIENTO DE LOS CLASIFICADORES

addpath('/Users/alejandralandinez/Documents/MATLAB/prtools/prtools')
[Data_training,PC_training,Ws_training,W_training,Ap_training] = entrenamiento(feature_training,class_training);
name_classf = {'nmc','knnc','ld','qdc','parzenc','dtc','neurc'};
[E,C]=testc(Ap_training*PC_training*W_training);
minE = min(E);
IminE = find(E==minE);
minimoerror=IminE(1);
nombremejorclasificador=name_classf(IminE);
%confmat(Ap_training*PC_training*W_training{minimoerror});

%% IV PARTE: EXTRACCIÓN DEL DATASET DE LOS DATOS DE PRUEBA

[Data_testing] = preparacion(feature_testing,class_testing);
 

%% V PARTE: CLASIFICACIÓN DE LOS DATOS DE PRUEBA

LABEL = labeld(Data_testing,Ws_training*PC_training*W_training{minimoerror});

error=zeros(1,length(class_testing));
for i=1:length(class_testing)
    if(class_testing(i)~=LABEL(i))
        error(i)=1;
    end
end
E=sum(error)/length(class_testing);
fprintf('El error de clasificación es: %d',E);