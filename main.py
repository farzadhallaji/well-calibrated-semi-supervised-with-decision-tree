import numpy

from weka.core.converters import Loader

from weka.core.dataset import Instances
from weka.core.dataset import Instance
from weka.classifiers import Evaluation
from weka.core.classes import Random

from weka.classifiers import Classifier
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

    fileOut.write("\n\nSelf-Training   data========"+str(1-eval.error_rate*100)+" number of instances=="+str(f3.num_instances)+"\n")
    fileOut.write("\n Error Rate=="+str(eval.error_rate) + "\n")

    fileOut.write("\n     precision   recall     areaUnderROC            \n\n");
    for i in range(test.get_instance(0).num_classes) :
        fileOut.write(str(eval.precision(i)) +"  "+str(eval.recall(i)) + "  "  +  str(eval.area_under_roc(i))+"\n")

    return eval




def LabeledUnlabeldata(data, unlabeled, tree , y) :
    
    data1 = Instances.copy_instances(data)
    labeling = Instances.copy_instances(unlabeled)
    tree.build_classifier(data1)
    
    j = i = s = l = 0

    # iris_data = loader.load_file(iris_file)

    while i < labeling.num_instances:
        clsLabel= tree.classify_instance(labeling.get_instance(i))
        dist = tree.distribution_for_instance(labeling.get_instance(i))

        # r=eval.getPrediction(tree, labeling.instance(i));
        # predict[i]=r.predicted();

        for k,dk in enumerate(dist) :
            if dk >= y :

                j=i
                while j < labeling.num_instances :
                    clsLabel= tree.classify_instance(labeling.get_instance(j))
                    # r=eval.getPrediction(tree, labeling.instance(i));
                    dist = tree.distribution_for_instance(labeling.get_instance(j))

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


datasaetsName = ["bupa"]
# datasaetsName = ["breast-cancer","bupa","colic","diabetes","heart","hepatitis","ionosphere","sick","sonar","tic-tac-toe","twomoons","votes"]
PathToData = "./originDataset/"

for dataset in datasaetsName :

    fileOut = open(PathToData+dataset+"/pyres-semi.txt", "w")

    fileOut.write(dataset+"\n")
    fileOut.write("\n")
    fileOut.write("########################################################");
    fileOut.write("\n")
    
    # for y in range(0.999,0.02,-0.001) :
    for y in numpy.arange(0.999,0.9,-0.001) :
        try:
            
            fileOut.write("##################### y =>  " + str(y)+"\n");

            data = loader.load_file(PathToData + dataset + "/train.arff")
            # f1 = loader.load_file(PathToData + dataset + "/train.arff")
            test =loader.load_file(PathToData + dataset + "/test.arff")
            data.class_is_last()
            test.class_is_last()
            # f1.class_is_last()

            labledDataSet , UnlabledDataSet = splitTrainSet(data);

            
            # tree = Classifier(classname="weka.classifiers.trees.J48", options=["-A"])
            tree = Classifier(classname="weka.classifiers.trees.J48")

            tree.build_classifier(labledDataSet)
            
            eval = Evaluation(labledDataSet)
            eval.test_model(tree, test)
            
            fileOut.write("Labeled data======== " + str(1.0 - eval.error_rate * 100) + " number of instances== " + str(labledDataSet.num_instances) + "\n")
            
            Newtrainpool = LabeledUnlabeldata(labledDataSet, UnlabledDataSet, tree, y)

            fileOut.write("Labeled data======== " + str(1.0 - eval.error_rate * 100) + " number of instances== " + str(labledDataSet.num_instances) + "\n")

            fileOut.write("           Decision Tree                       \n")
            fileOut.write("\n      precision   recall     areaUnderROC            \n\n")

            for i in range(test.get_instance(0).num_classes) :
                fileOut.write(str(eval.precision(i)) +"  "+str(eval.recall(i)) + "  "  +  str(eval.area_under_roc(i))+"\n")



            ClassifyWithDT(Newtrainpool, test, tree, fileOut)

            fileOut.write("\n")
            fileOut.write("########################################################\n")
            fileOut.write("\n")
            
        except Exception as e:
            raise e

    fileOut.write("\n")
    fileOut.write("\n")
    fileOut.write("########################################################\n")
    fileOut.write("########################################################\n")
    fileOut.write("########################################################\n")
    fileOut.write("\n")
    fileOut.write("\n")


jvm.stop()



