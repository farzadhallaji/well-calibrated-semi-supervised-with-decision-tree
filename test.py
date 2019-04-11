from weka.core.converters import Loader

from weka.core.dataset import Instances
from weka.core.dataset import Instance
from weka.classifiers import Evaluation


from weka.classifiers import Classifier
# cls = Classifier(classname="weka.classifiers.trees.J48")
# cls = Classifier(classname="weka.classifiers.evaluation")

from sklearn.linear_model import LogisticRegression as LR
from sklearn.isotonic import IsotonicRegression as IR

import numpy


# iris_data = loader.load_file(iris_file)
# iris_data.class_is_last()


import weka.core.jvm as jvm
import os
os.environ["_JAVA_OPTIONS"] = "-Dfile.encoding=UTF-8"
jvm.start(packages=True)

loader = Loader("weka.core.converters.ArffLoader")

instances = loader.load_file("/home/farzad/Desktop/jrnl/semiSupervisedPython/originDataset/bupa/train.arff")
instances.class_is_last()

tree = Classifier(classname="weka.classifiers.trees.J48")

tree.build_classifier(instances)
# clsLabel = j48.classify_instance(data.get_instance(0))
# print("====================================>",clsLabel)






p_train = numpy.zeros(shape=(instances.num_instances,1))
y_train = numpy.zeros(shape=(instances.num_instances,1))

for i,instance in enumerate(instances) :
    dist = tree.distribution_for_instance(instance)
    p_train[i] = [ dist[1]  ]
    y_train[i] = [tree.classify_instance(instance)]



print("p_train ======> > > >>>> > > >>>> " , len(p_train))
print("p_train ======> > > >>>> > > >>>> " , len(y_train))
print("p_train ======> > > >>>> > > >>>> " , instances.num_instances)
# print("p_train ======> > > >>>> > > >>>> " , p_train)
# print("p_train ======> > > >>>> > > >>>> " , p_train.reshape( -1, 1 ))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

lr = LR(solver='lbfgs')                                                      
lr.fit( p_train , numpy.ravel(y_train,order='C') )     # LR needs X to be 2-dimensional
# lr.fit( p_train.reshape( -1, 1 ), y_train )     # LR needs X to be 2-dimensional

dist = tree.distribution_for_instance(instances.get_instance(80))[1]
tmp = numpy.zeros(shape=(1,1))
tmp[0] = [dist]
# print(dist)
print("lr.predict_proba( dist.reshape( -1, 1 ))[:,1] ====>>>>>" , instances)
# print("lr.predict_proba( dist.reshape( -1, 1 ))[:,1] ====>>>>>" , dist)
# print("lr.predict_proba( dist.reshape( -1, 1 ))[:,1] ====>>>>>" , instances.get_instance(20))
# print("lr.predict_proba( dist.reshape( -1, 1 ))[:,1] ====>>>>>" , lr.predict(tmp.reshape(1, -1)))
# print("lr.predict_proba( dist.reshape( -1, 1 )) ====>>>>>" , lr.predict_proba( tmp.reshape(1, -1)))


ir = IR( out_of_bounds = 'clip' )
ir.fit(numpy.ravel(p_train,order='C')  , numpy.ravel(y_train,order='C')  )
print("ir.transform( tmp.reshape(1, -1)) =========>>>>>>   ",ir.transform( numpy.ravel(tmp,order='C'))[0])
# print("ir.transform( tmp.reshape(1, -1)) =========>>>>>>   ", numpy.ravel(tmp,order='C') )

for i in lr.predict_proba( tmp.reshape(1, -1))[0] :
    print(i,"\n")



jvm.stop()

