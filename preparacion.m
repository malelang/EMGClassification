%% ESTA FUNCIÓN HACE LA PREPARACIÓN DE LOS DATOS DE TESTING EN UN PRDATASET
function [DatasetTotal]=preparacion(feature_testing,class_testing)

lista_feasures = char('Vrms_E1','Vrms_E2','Vrms_E3','Vrms_E4','Vrms_E5','Vrms_E6','Vrms_E7','Vrms_E8','Coef9','Coef10','Coef11','Coef12','Coef13','Coef14','Coef15','Coef16','Coef17','Coef18','Coef19','Coef20','Coef21','Coef22','Coef23','Coef24','Coef25','Coef26','Coef27','Coef28','Coef29','Coef30','Coef31','Coef32','Coef33','Coef34','Coef35','Coef36','Coef37','Coef38','Coef39','Coef40','mav1','mav2','mav3','mav4','mav5','mav6','mav7','mav8','zc1','zc2','zc3','zc4','zc5','zc6','zc7','zc8','ssc1','ssc2','ssc3','ssc4','ssc5','ssc6','ssc7','ssc8');
DatasetTotal = prdataset(feature_testing, class_testing,'featlab',lista_feasures);
end