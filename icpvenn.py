import warnings

from weka.core.converters import Loader
from weka.core.dataset import Instances
from weka.core.dataset import Instance
from weka.classifiers import Evaluation
from weka.core.classes import Random
from weka.classifiers import Classifier

from sklearn.linear_model import LogisticRegression as LR

from sklearn.isotonic import IsotonicRegression as IR

import numpy as np
from sklearn.svm import SVC
from nonconformist.cp import IcpClassifier
from nonconformist.nc import NcFactory


# cls = Classifier(classname="weka.classifiers.trees.J48", options=["-A"])
# cls = Classifier(classname="weka.classifiers.evaluation")



# iris_data = loader.load_file(iris_file)
# iris_data.class_is_last()



def splitTrainSet(data,m_numLabledData=10) :

    total = data.num_instances
    labeled_amount = int(m_numLabledData * total / 100)
    unlabeled_amount = total - labeled_amount

    rand = Random(1)
    data.randomize(rand)

    labledDataSet = Instances.create_instances(data.relationname,data.attributes(),labeled_amount)
    UnlabledDataSet = Instances.create_instances(data.relationname,data.attributes(),unlabeled_amount)

    

    for i in range(labeled_amount) :
        labledDataSet.add_instance(data.get_instance(i))

    labledDataSet.randomize(rand)

    for i in range(unlabeled_amount) :
        UnlabledDataSet.add_instance(data.get_instance(labeled_amount + i))


    UnlabledDataSet.no_class()
    UnlabledDataSet.randomize(rand)

    labledDataSet.class_is_last()
    UnlabledDataSet.class_is_last()


    return labledDataSet,UnlabledDataSet


def ClassifyWithDT(f3, test, tree , fileOut) :

    eval= Evaluation(f3)
    tree.build_classifier(f3)

    eval.test_model(tree, test)

    print ("\n\nSelf-Training   data========"+str((1-eval.error_rate)*100)+" number of instances=="+str(f3.num_instances)+"\n")
    print ("\n Error Rate=="+str(eval.error_rate) + "\n")

    print ("\n     precision   recall     areaUnderROC            \n\n");
    for i in range(test.get_instance(0).num_classes) :
        print (str(eval.precision(i)) +"  "+str(eval.recall(i)) + "  "  +  str(eval.area_under_roc(i))+"\n")

    return eval


def calculate_probability_distribution(tree , instances , index , cal_method =None):

	if cal_method == None :
		return tree.distribution_for_instance(instances.get_instance(index))

	elif cal_method == 'Platt' :

		p_train = np.zeros(shape=(instances.num_instances,1))
		y_train = np.zeros(shape=(instances.num_instances,1))

		for i,instance in enumerate(instances) :
		    dist = tree.distribution_for_instance(instance)
		    p_train[i] = [ (dist[1] - 0.5)*2.0 ]
		    y_train[i] = [instance.get_value(instance.class_index)]

		# print("p_train ====>>>" , p_train)
		# print("y_train ====>>>" , y_train)

		dist = (tree.distribution_for_instance(instances.get_instance(index))[1]-0.5)*2.0
		tmp = np.zeros(shape=(1,1))
		tmp[0] = [dist]

		print(np.sum(y_train))
		if np.sum(y_train) in [len(y_train),0]:
			print("all one class")
			for ins in instances : 
				print("ins ===> " , ins)
			return tree.distribution_for_instance(instances.get_instance(index))

		else :

			warnings.filterwarnings("ignore", category=FutureWarning)
			lr = LR(solver='lbfgs')                                                      
			lr.fit( p_train , np.ravel(y_train,order='C') )

			return lr.predict_proba( tmp.reshape(1, -1))[0]


	elif cal_method == 'Isotonic' :

		p_train = np.zeros(shape=(instances.num_instances,1))
		y_train = np.zeros(shape=(instances.num_instances,1))

		for i,instance in enumerate(instances) :
		    dist = tree.distribution_for_instance(instance)
		    p_train[i] = [ dist[1] ]
		    y_train[i] = [instance.get_value(instance.class_index)]


		dist = tree.distribution_for_instance(instances.get_instance(index))[1]
		tmp = np.zeros(shape=(1,1))
		tmp[0] = [dist]

		print(np.sum(y_train))
		if np.sum(y_train) in [len(y_train),0]:
			print("all one class")
			for ins in instances : 
				print("ins ===> " , ins)
			return tree.distribution_for_instance(instances.get_instance(index))

		else :

			ir = IR( out_of_bounds = 'clip' )
			ir.fit(np.ravel(p_train,order='C')  , np.ravel(y_train,order='C'))

			p = ir.transform( np.ravel(tmp,order='C'))[0]
			return [p,1-p]
			
	# elif cal_method == 'ProbabilityCalibrationTree' :
	# 	pass


	elif cal_method == 'ICP' :


		pass
	elif cal_method == 'Venn' :
		pass


def LabeledUnlabeldata(data, unlabeled, tree, y, cal_method=None ) :
    
    data1 = Instances.copy_instances(data)
    labeling = Instances.copy_instances(unlabeled)
    tree.build_classifier(data1)
    
    j = i = s = l = 0

    while i < labeling.num_instances:
        clsLabel= tree.classify_instance(labeling.get_instance(i))

        ##### probability calculation #####
        # dist = tree.distribution_for_instance(labeling.get_instance(i))
        dist = calculate_probability_distribution(tree , labeling , i , cal_method)


        for k,dk in enumerate(dist) :
            if dk >= y :

                j=i
                while j < labeling.num_instances :
                    clsLabel= tree.classify_instance(labeling.get_instance(j))

                    ##### probability calculation #####
                    # dist = tree.distribution_for_instance(labeling.get_instance(j))
                    dist = calculate_probability_distribution(tree , labeling , j , cal_method)

                    for dp in dist :
                        if dp >= y :
                            inst = labeling.get_instance(i)
                            inst.set_value(inst.class_index,clsLabel)
                            data1.add_instance(inst)
                            labeling.delete(i)
                            l+=1
                            j-=1

                    j+=1

            if k==(len(dist)-1) and (l!=0) :
                    tree.build_classifier(data1)
                    i=-1
                    s+=l
                    l=0
        i+=1

    data1.compactify()
    return data1





import weka.core.jvm as jvm
import os
os.environ["_JAVA_OPTIONS"] = "-Dfile.encoding=UTF-8"
jvm.start(packages=True)

loader = Loader("weka.core.converters.ArffLoader")

Method = "Platt"


# datasaetsName = ["bupa"]
datasaetsName = ["breast-cancer","bupa","colic","diabetes","heart","hepatitis","ionosphere","sick","sonar","tic-tac-toe","twomoons","votes"]
PathToData = "./originDataset/"

for dataset in datasaetsName :

    fileOut = open(PathToData+dataset+"/pyres-"+Method+".txt", "w")

    print (dataset+"\n")
    print(dataset+"\n")
    print ("\n")
    print ("########################################################\n");
    print ("\n")
    
    # for y in range(0.999,0.02,-0.001) :
    for y in np.arange(0.999,0.99,-0.001) :
        try:
            
            print ("##################### y =>  " + str(y)+"\n");
            print("##################### y =>  " + str(y)+"\n");

            data = loader.load_file(PathToData + dataset + "/train.arff")
            # f1 = loader.load_file(PathToData + dataset + "/train.arff")
            test =loader.load_file(PathToData + dataset + "/test.arff")
            data.class_is_last()
            test.class_is_last()
            # f1.class_is_last()

            labledDataSet , UnlabledDataSet = splitTrainSet(data);

            
            tree = Classifier(classname="weka.classifiers.trees.J48", options=["-A"])
            # tree = Classifier(classname="weka.classifiers.trees.J48")

            tree.build_classifier(labledDataSet)
            
            eval = Evaluation(labledDataSet)
            eval.test_model(tree, test)
            
            print ("Labeled data======== " + str((1.0 - eval.error_rate )* 100) + " number of instances== " + str(labledDataSet.num_instances) + "\n")
            
            Newtrainpool = LabeledUnlabeldata(labledDataSet, UnlabledDataSet, tree, y , cal_method=Method)

            print ("\n\nLabeled data======== " + str((1.0 - eval.error_rate )* 100) + " number of instances== " + str(labledDataSet.num_instances) + "\n")

            print ("           Decision Tree                       \n")
            print ("\n      precision   recall     areaUnderROC            \n\n")

            for i in range(test.get_instance(0).num_classes) :
                print (str(eval.precision(i)) +"  "+str(eval.recall(i)) + "  "  +  str(eval.area_under_roc(i))+"\n")



            ClassifyWithDT(Newtrainpool, test, tree, fileOut )

            print ("\n")
            print ("########################################################\n")
            print ("\n")
            
        except Exception as e:
            raise e

    print ("\n")
    print ("\n")
    print ("########################################################\n")
    print ("########################################################\n")
    print ("########################################################\n")
    print ("\n")
    print ("\n")


jvm.stop()
