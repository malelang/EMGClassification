function [DatasetTotal,PC,Ws,W,Ap]=entrenamiento(feature_training,class_training)
%% ENTRENAMIENTO DE LOS CLASIFICADORES
%Variables de salida:
%Dataset Total: lista de los datos junto con las características y las 
%clases de cada registro, para los datos de entrenamiento
%PC: componentes principales
%Ws: Atributos seleccionados
%W: Conjunto de clasificadores
%E: Error mínimo

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
randreset;
[Ae,Ap,IL,IU] = gendat(DatasetTotal,0.75); 
figure
scatterd(DatasetTotal);
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
%w7 = svc(Ae*PC,proxm('p',6));
w8 = neurc(Ae*PC);      % Neuronal
%w9 = adaboostc(Ae*PC,perlc([],1),50,[],1); %AdaBoost

% Compute and display errors
% Store classifiers in a cell
W = {w1,w2,w3,w4,w5,w6,w8};
name_classf = {'nmc','knnc','ld','qdc','parzenc','dtc','neurc'};
[E,C]=testc(Ap*PC*W); % E, error, C{i} # errores de clasificaci�n en cada clase. Ver tambien confmat  
disp('Error de los clasificadores')
disp(E)
disp('Error minimo')
minE = min(E);
disp(minE)
IminE = find(E==minE);
disp('Mejores Clasificadores')  % Nombres de los clasificadores con m�nimo error de clasificaci�n
nombremejorclasificador=name_classf(IminE);
disp(nombremejorclasificador)


end