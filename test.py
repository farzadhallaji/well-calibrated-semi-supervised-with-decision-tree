from weka.core.converters import Loader

from weka.core.dataset import Instances
from weka.core.dataset import Instance
from weka.classifiers import Evaluation


from weka.classifiers import Classifier
# cls = Classifier(classname="weka.classifiers.trees.J48")
# cls = Classifier(classname="weka.classifiers.evaluation")



# iris_data = loader.load_file(iris_file)
# iris_data.class_is_last()


import weka.core.jvm as jvm
import os
os.environ["_JAVA_OPTIONS"] = "-Dfile.encoding=UTF-8"
jvm.start(packages=True)

loader = Loader("weka.core.converters.ArffLoader")

data = loader.load_file("anneal.arff")
data.class_is_last()

j48 = Classifier(classname="weka.classifiers.trees.J48")

j48.build_classifier(data)

clsLabel = j48.classify_instance(data.get_instance(0))

print("====================================>",clsLabel)



jvm.stop()







def LabeledUnlabeldata(data, unlabeled, tree , y) :
    
    data1 = Instances.copy_instances(data)
    labeling = Instances.copy_instances(unlabeled)
    tree.build_classifier(data1)
    
    j = i = s = l = 0

    # iris_data = loader.load_file(iris_file)

    while i < labeling.num_instances:
        clsLabel= tree.classify_instance(labeling.get_instance(i))
        dist = cls.distribution_for_instance(labeling.get_instance(i))

        # r=eval.getPrediction(tree, labeling.instance(i));
        # predict[i]=r.predicted();

        for dk in dist :
            if dk >= y :

                j=i
                while j < labeling.num_instances :
                    clsLabel= tree.classify_instance(labeling.get_instance(j))
                    # r=eval.getPrediction(tree, labeling.instance(i));
                    dist = cls.distribution_for_instance(labeling.get_instance(j))

                    for dp in dist :
                        if dp >= y :
                            inst = labeling.get_instance(i)
                            inst.set_value(inst.class_index,clsLabel)



                            pass

        pass

    # while (i<labeling.numInstances()){
    #         clsLabel= tree.classifyInstance(labeling.instance(i));
    #         r=eval.getPrediction(tree, labeling.instance(i));
    #         double[] dist=tree.distributionForInstance(labeling.instance(i));
    #         predict[i]=r.predicted();
    #         for (int k=0; k<dist.length; k++){
    #             if(dist[k]>=y){



    #                 j=i;
    #                 while(j<labeling.numInstances()){

    #                     clsLabel= tree.classifyInstance(labeling.instance(j));
    #                     r=eval.getPrediction(tree, labeling.instance(j));
                        # dist=tree.distributionForInstance(labeling.instance(j));
                        # for (int p=0; p<dist.length; p++){
                            # if(dist[p]>=y){
                                # //   System.out.println("tanhan="+j+"\n");
                                labeling.instance(j).setClassValue(clsLabel);
                                Instance x = labeling.instance(j);
                                //   System.out.println("\n X= "+x);
                                data1.add(x);
                                labeling.delete(j);
                                l++;
                                j=j-1;
                            }// if
                        }// for
                        //    predict[j]=r.predicted();
                        j++;
                    }// while
                }// if
                if( (k==(dist.length-1)) && (l!=0) ){
                    tree.setUseLaplace(true);
                    tree.buildClassifier(data1);
                    i=-1; s=s+l; l=0;
                }
            }// for
            i++;
        }// while

        data1.compactify();
        return data1;

    }
"""