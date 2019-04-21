PathToData = ""

datasaetsName = ["breast-cancer"]
# datasaetsName = ["breast-cancer","bupa","colic","diabetes","heart","hepatitis","ionosphere","sick","sonar","tic-tac-toe","twomoons","votes"]

for iii in range(len(datasaetsName)):
	path = PathToData+datasaetsName[iii]+"/pyres-Platt.txt"
	# print(path)
	max_precision = [0.0 , 0.0]
	max_recall = [0.0 , 0.0]
	max_areaUnderROC = [0.0 , 0.0]
	resultlines=''
	max_labeled = 0
	with open(path, "r") as f:
		plain = f.read()

		reses = plain.split("########################################################")
		

		# print(len(reses))
		# print(reses[0].split("\n")[2])
		cnt = 0
		for rest in reses :
			lines = rest.split("\n")
			# if len(lines) < 5 : 
			# 	print("zzzzzit")
			# try:
			if len(lines) == 25 :
				
				labeled = lines[15].split("==")
				labeled = int(labeled[5])

				l1 = lines[11].split(" ")
				l2 = lines[12].split(" ")
				lr1 = lines[21].split(" ")
				lr2 = lines[22].split(" ")

				precision = []
				precisionR = []
				precision.append(float(l1[0]))
				precision.append(float(l2[0]))
				precisionR.append(float(lr1[0]))
				precisionR.append(float(lr2[0]))
				# print(precisionR)
				# print(l1)
				
				recall = []
				recallR = []
				recall.append(float(l1[2]))
				recall.append(float(l2[2]))
				recallR.append(float(lr1[2]))
				recallR.append(float(lr2[2]))

				areaUnderROC = []
				areaUnderROCR = []
				areaUnderROC.append(float(l1[4]))
				areaUnderROC.append(float(l2[4]))
				areaUnderROCR.append(float(lr1[4]))
				areaUnderROCR.append(float(lr2[4]))

				if precisionR[0] > max_precision[0] or precisionR[1] > max_precision[1] :
					max_precision = precisionR
					max_recall = recallR
					max_areaUnderROC = areaUnderROCR
					resultlines = lines
					max_labeled = labeled

				elif precisionR[0] == max_precision[0] or precisionR[1] == max_precision[1] :
					if labeled > max_labeled :
						max_precision = precisionR
						max_recall = recallR
						max_areaUnderROC = areaUnderROCR
						resultlines = lines
						max_labeled = labeled
				else :
					if precisionR[0] == max_precision[0] and precisionR[1] == max_precision[1] :
						if recallR[0] > max_recall[0] or recallR[1] > max_recall[1] or areaUnderROCR[0] > max_areaUnderROC[0] or areaUnderROCR[1] > max_areaUnderROC[1] :
							max_precision = precisionR
							max_recall = recallR
							max_areaUnderROC = areaUnderROCR
							resultlines = lines
							max_labeled = labeled


				# print(cnt)
				# print (lines[cnt])
				# cnt +=1 
		
			f.close()
	print(max_precision)

	with open(PathToData+datasaetsName[iii]+"/pyres_platt.txt", "w") as out:
		for item in resultlines:
			out.write("%s\n" % item)

		out.close()

				# print( labeled )
				# pass
			# except Exception as e:
				# raise e
				# pass