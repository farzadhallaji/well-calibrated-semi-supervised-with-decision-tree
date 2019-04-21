from weka.core.converters import Loader
from weka.core.dataset import Instances
from weka.core.dataset import Instance
from weka.classifiers import Evaluation
from weka.core.classes import Random
from weka.classifiers import Classifier


import weka.core.jvm as jvm
import os
os.environ["_JAVA_OPTIONS"] = "-Dfile.encoding=UTF-8"
jvm.start(packages=True)

loader = Loader("weka.core.converters.ArffLoader")



datasaetsName = ["bupa"]
# datasaetsName = ["breast-cancer","bupa","colic","diabetes","heart","hepatitis","ionosphere","sick","sonar","tic-tac-toe","twomoons","votes"]
PathToData = "./originDataset/"

data = loader.load_file(PathToData + datasaetsName[0] + "/train.arff")
test =loader.load_file(PathToData + datasaetsName[0] + "/test.arff")
data.class_is_last()
test.class_is_last()

# tree = Classifier(classname="weka.classifiers.trees.J48", options=["-A"])
# tree.build_classifier(data)

print(data.get_instance(0))
print(data.get_instance(1))
print(data.get_instance(2))
print(data.get_instance(3))
print(data.get_instance(4))
print(data.get_instance(5))
print(data.delete(2))
print(data.delete(3-1))
print(data.delete(4-2))
print(data.get_instance(0))
print(data.get_instance(1))
print(data.get_instance(2))
print(data.get_instance(3))
print(data.get_instance(4))
print(data.get_instance(5))
# eval = Evaluation(data)
# eval.test_model(tree, test)


# print("\n\nLabeled data======== " + str((1.0 - eval.error_rate )* 100) + " number of instances== " + str(data.num_instances) + "\n")

# print("           Decision Tree                       \n")
# print("\n      precision   recall     areaUnderROC            \n\n")

# for i in range(test.get_instance(0).num_classes) :
#     print(str(eval.precision(i)) +"  "+str(eval.recall(i)) + "  "  +  str(eval.area_under_roc(i))+"\n")


jvm.stop()